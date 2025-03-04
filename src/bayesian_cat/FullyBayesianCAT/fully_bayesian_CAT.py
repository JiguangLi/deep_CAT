import typing
import numpy as np
from .bayesian_test_taker import BayesianTestTakers
from tqdm import tqdm
from scipy.stats import norm
import time

class FullyBayesianCAT:

    def __init__(
            self,
            true_thetas: np.ndarray,
            test_bank: typing.Dict[str, np.ndarray],
            selection_criterion: str,
            max_items: int,
            mc_samples: int,
            plugin_response: bool = False,
            true_response: np.ndarray = None,
            starting_item: typing.Optional[int] = None,
            sir_params: typing.Dict[str, typing.Any] = {},
            num_workers: int = 8,
            random_state: int = 42,
    ):
        """
        Create a Bayesian Computerized Testing Object for Simulation

            Args:
                true_thetas: Shape: (n, k), latent factors of new test takers
                test_bank: dictionary with loading matrix "alphas", and "intercepts" for the items in the test bank
                selection_criterion: kl_eap, kl_pos, mi, or "mi_sir"
                max_items: maximum numbers to test, currently no stopping criterions for the simulation
                mc_samples: number of mc samples
                starting_item: first item to start or we do self select
                sir_params: dictionary containing the parameters of sampling-resampling procedure (posterior reweighting)
                num_workers: number of cores for multiprocessing
                random_state: seed
            Raises:
                ValueError: if given inputs do not meet object expectations.
        """
        self.seed = random_state
        self.true_thetas = true_thetas
        self.n = true_thetas.shape[0]
        self.k = true_thetas.shape[1]
        self.alphas = test_bank["alphas"] # samples * m * k
        self.intercepts = test_bank["intercepts"] # samples * m
        self.pos_alphas = np.mean(self.alphas, axis=0)
        self.pos_intercepts = np.mean(self.intercepts, axis = 0)
        self.pos_num_samples = self.alphas.shape[0]
        self.sc = selection_criterion
        self.max_items = max_items
        self.mc_samples = mc_samples
        self.num_workers = num_workers
        self.rng = np.random.default_rng(random_state)
        self.m = self.alphas.shape[1]
        self.item_selections = np.zeros((self.n, self.max_items))
        self.plugin_response = plugin_response
        self.response = true_response
        if starting_item:
            self.start = starting_item
        else:
            self.start = self.pick_starting_item()
        if self.sc == "mi_sir" and len(sir_params) == 0:
            raise ValueError("No sampling-resampling (sir_params) parameter available for mi_sir")
        elif self.sc == "mi_sir" or self.sc == "kl_oracle" or self.sc == "variance_reduction":
            self.sir_start = sir_params["sir_start"]
            self.sir_large_samples, self.sir_samples = sir_params["large_samples"], sir_params["samples"]
        self.tts = BayesianTestTakers(thetas=self.true_thetas, pos_num_samples = self.pos_num_samples,
                                      max_item=self.max_items, num_workers=self.num_workers,
                                      random_state=self.seed)

    def pick_starting_item(self):
        """Pick the starting item from the test bank

        Currently, consider picking item with around median intercept, and relatively high magnitude of alpha parameters
        """
        icpt_40 = np.quantile(self.pos_intercepts, 0.4)
        icpt_60 = np.quantile(self.pos_intercepts, 0.6)
        indices = np.where((self.pos_intercepts >= icpt_40) & (self.pos_intercepts <= icpt_60))[0]
        abs_sum = np.sum(np.abs(self.pos_alphas[indices]), axis=1)
        upper_quantile_value = np.quantile(abs_sum, 0.75)
        arg_max_upper_quantile = np.argmin(np.abs(abs_sum - upper_quantile_value))
        return indices[arg_max_upper_quantile]

    def simulate(self):
        """For each test taker, simulate their responses to selected items and record theta_updates"""
        # starting
        start_items = np.array([self.start]*self.n)
        selected_alphas = self.alphas[:, start_items, :] # s * n * k
        selected_intercepts = self.intercepts[:, start_items] # s * n
        if self.plugin_response:
            true_responses = self.response[np.arange(self.response.shape[0]), start_items]
            self.tts.answer_true_responses(selected_alphas, selected_intercepts, start_items,
                                           response_vec=true_responses.astype(int))
        else:
            self.tts.answer_items(selected_alphas, selected_intercepts, start_items)
        # simulate
        for j in tqdm(range(1, self.max_items)):
            new_items = self.select_items(j)
            selected_alphas = self.alphas[:, new_items, :]  # s * n * k
            selected_intercepts = self.intercepts[:, new_items]  # s * n
            if self.plugin_response:
                true_responses = self.response[np.arange(self.response.shape[0]), new_items]
                self.tts.answer_true_responses(selected_alphas, selected_intercepts, new_items,
                                               response_vec=true_responses.astype(int))
            else:
                self.tts.answer_items(selected_alphas, selected_intercepts, new_items)
        self.item_selections = self.tts.item_ids

    def select_kl_eap(self, item_idx: int):
        """Select Items Based on Posterior Expected KL Information"""
        result = np.zeros(self.n, dtype=int)
        new_samples, pos_mean = self.tts.get_pos_mean(self.mc_samples)
        for i in range(self.n):
            theta_hat = pos_mean[i].reshape(-1, 1)
            theta_sampled = new_samples[i] # pos_samples * mc_samples by k
            avail_items = np.setdiff1d(np.arange(self.m), self.tts.item_ids[i, :item_idx])
            cur_alphas = self.alphas[:, avail_items, :] # pos_samples * a * k
            cur_intercepts = self.intercepts[:, avail_items] # pos_samples * a
            p_hat = np.mean(norm.cdf(cur_alphas @ theta_hat.flatten() + cur_intercepts), axis=0) # j-dim vec
            p_sampled = np.einsum('pak,pmk->pam', cur_alphas, theta_sampled) # pos_samples by a by m
            p_sampled += cur_intercepts[:, :, np.newaxis]
            p_sampled = np.transpose(p_sampled, axes = (1,0,2)) # a by p_samples by m
            p_sampled = p_sampled.reshape(len(avail_items), self.mc_samples*self.pos_num_samples) # a by (p*m)
            p_sampled = norm.cdf(p_sampled)
            kl_info = p_hat * np.mean(
                np.log(1/p_sampled * p_hat.reshape(-1, 1)), axis=1) + (1-p_hat) * np.mean(
                np.log(1/(1-p_sampled) * (1-p_hat).reshape(-1, 1)), axis=1)
            result[i] = avail_items[np.argmax(kl_info)]
        return result

    def select_kl_pos(self, item_idx: int):
        """Select Items Based on KL Distance Between Subsequent Posteriors"""
        result = np.zeros(self.n, dtype=int)
        all_items = np.arange(self.m)
        new_samples, pos_mean = self.tts.get_pos_mean(self.mc_samples)
        for i in range(self.n):
            theta_sampled = new_samples[i]  # pos_samples * mc_samples * k
            avail_items = all_items[~np.in1d(all_items, self.tts.item_ids[i, :item_idx])]
            cur_alphas = self.alphas[:, avail_items, :]  # pos_samples * a * k
            cur_intercepts = self.intercepts[:, avail_items]  # pos_samples * a
            p_sampled = np.einsum('pak,pmk->pam', cur_alphas, theta_sampled)  # pos_samples by a by m
            p_sampled += cur_intercepts[:, :, np.newaxis]
            p_sampled = np.transpose(p_sampled, axes=(1, 0, 2))  # a by p_samples by m
            p_sampled = p_sampled.reshape(len(avail_items), self.mc_samples * self.pos_num_samples)  # a by (p*m)
            p_sampled = norm.cdf(p_sampled)
            pred_hat = np.mean(p_sampled, axis=1)
            kl_term1 = pred_hat * np.mean(np.log(1 / p_sampled * pred_hat.reshape(-1, 1)), axis=1)
            kl_term2 = (1 - pred_hat) * np.mean(np.log(1 / (1 - p_sampled) * (1 - pred_hat).reshape(-1, 1)), axis=1)
            result[i] = avail_items[np.argmax(kl_term1 + kl_term2)]
        return result

    def select_mi_sir(self, item_idx: int):
        """Select Items Based on Mutual Information, but use posterior reweighting to accelerate"""
        result = np.zeros(self.n, dtype=int)
        all_items = np.arange(self.m)
        new_samples, pos_mean = self.tts.get_pos_mean(self.mc_samples)
        for i in range(self.n):
            theta_sampled = new_samples[i] # pos_samples * mc_samples * k
            avail_items = all_items[~np.in1d(all_items, self.tts.item_ids[i, :item_idx])]
            cur_alphas = self.alphas[:, avail_items, :]  # pos_samples * a * k
            cur_intercepts = self.intercepts[:, avail_items]  # pos_samples * a
            reweighted_samples = self.tts.get_sir_reweighted_samples(theta_sampled, self.sir_samples,
                                                                     cur_alphas,
                                                                     cur_intercepts)
            pred_hat = reweighted_samples["pred_hat"]
            samples0, samples1 = reweighted_samples["0"], reweighted_samples["1"] # a by pos_samples * mc_samples * k
            mi_vals = np.zeros(samples0.shape[0])
            for l in range(samples0.shape[0]):
                cur_thetas0, cur_thetas1= samples0[l], samples1[l] # p by mc by k
                # p sampled 1
                p_sampled1 = np.einsum('pmk,pk->pm', cur_thetas1, cur_alphas[:, l, :]) # pos_samples * m
                d_temp = cur_intercepts[:, l]
                p_sampled1 += d_temp[:, np.newaxis] # p by m
                p_sampled1 = norm.cdf(p_sampled1) # p by m
                # p sampled 0
                p_sampled0 = np.einsum('pmk,pk->pm', cur_thetas0, cur_alphas[:, l, :])  # pos_samples * m
                p_sampled0 += d_temp[:, np.newaxis]  # p by m
                p_sampled0 = norm.cdf(p_sampled0)  # p by m
                p_sampled0 = 1 - p_sampled0
                mi_term1 = pred_hat[l] * np.mean(np.log(p_sampled1 / pred_hat[l]))
                mi_term2 = (1 - pred_hat[l]) * np.mean(np.log(p_sampled0 / (1 - pred_hat[l])))
                mi_vals[l] = mi_term1 + mi_term2
            result[i] = avail_items[np.argmax(mi_vals)]
        return result

    def select_var_e_pred(self, item_idx):
        """Select Items with the largest Predictive varainces around its mean int (p-c)^2 """
        result = np.zeros(self.n, dtype=int)
        all_items = np.arange(self.m)
        new_samples, pos_mean = self.tts.get_pos_mean(self.mc_samples)
        for i in range(self.n):
            theta_sampled = new_samples[i]  # pos_samples * mc_samples * k
            avail_items = all_items[~np.in1d(all_items, self.tts.item_ids[i, :item_idx])]
            cur_alphas = self.alphas[:, avail_items, :]  # pos_samples * a * k
            cur_intercepts = self.intercepts[:, avail_items]  # pos_samples * a
            p_sampled = np.einsum('pak,pmk->pam', cur_alphas, theta_sampled)  # pos_samples by a by m
            p_sampled += cur_intercepts[:, :, np.newaxis]
            p_sampled = np.transpose(p_sampled, axes=(1, 0, 2))  # a by p_samples by m
            p_sampled = p_sampled.reshape(len(avail_items), self.mc_samples * self.pos_num_samples)  # a by (p*m)
            p_sampled = norm.cdf(p_sampled)
            pred_var = np.var(p_sampled, axis=1)
            result[i] = avail_items[np.argmax(pred_var)]
        return result

    def select_items(self, item_idx):
        """Select Next Items Depending on Given Selection Criterion for each test taker"""
        if self.sc == "kl_eap":
            return self.select_kl_eap(item_idx)
        elif self.sc == "kl_pos":
            return self.select_kl_pos(item_idx)
        elif self.sc == "mi":
            return self.select_mi(item_idx)
        elif self.sc == "mi_sir":
            if item_idx >= self.sir_start:
                return self.select_mi_sir(item_idx)
            else:
                return self.select_mi(item_idx)
        elif self.sc == "predictive_variance_e":
            return self.select_var_e_pred(item_idx)
        else:
            raise ValueError("unknown selection criterion")

    def factor_inference(self, num_samples: int, ci: float =0.95):
        """Get Posterior Means and Confidence Intervals for the Final Estimates"""
        samples = self.tts.get_factor_samples(num_samples) # (n, num_samples, k)
        pos_mean = np.mean(samples, axis=1) # n by k
        ci_lb = np.quantile(samples, (1-ci)/2, axis=1)
        ci_ub = np.quantile(samples, ci + (1-ci)/2, axis=1)
        return {"pos_mean": pos_mean, "ci_lb": ci_lb, "ci_ub": ci_ub, "samples": samples}

















