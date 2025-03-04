import typing
import numpy as np
from .test_taker import TestTakers
from tqdm import tqdm
from scipy.stats import norm
import time

class BayesianCAT:

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
        self.alphas = test_bank["alphas"]
        self.intercepts = test_bank["intercepts"]
        self.sc = selection_criterion
        self.max_items = max_items
        self.mc_samples = mc_samples
        self.num_workers = num_workers
        self.rng = np.random.default_rng(random_state)
        self.m = self.alphas.shape[0]
        self.theta_updates = np.zeros((self.max_items, self.n, self.k))
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
        self.tts = TestTakers(thetas=self.true_thetas, max_item=self.max_items, num_workers=self.num_workers,
                              random_state=self.seed)

    def pick_starting_item(self):
        """Pick the starting item from the test bank

        Currently, consider picking item with around median intercept, and relatively high magnitude of alpha parameters
        """
        icpt_40 = np.quantile(self.intercepts, 0.4)
        icpt_60 = np.quantile(self.intercepts, 0.6)
        indices = np.where((self.intercepts >= icpt_40) & (self.intercepts <= icpt_60))[0]
        abs_sum = np.sum(np.abs(self.alphas[indices]), axis=1)
        upper_quantile_value = np.quantile(abs_sum, 0.75)
        arg_max_upper_quantile = np.argmin(np.abs(abs_sum - upper_quantile_value))
        return indices[arg_max_upper_quantile]

    def simulate(self):
        """For each test taker, simulate their responses to selected items and record theta_updates"""
        start_items = np.array([self.start]*self.n)
        if self.plugin_response:
            true_responses = self.response[np.arange(self.response.shape[0]), start_items]
            self.tts.answer_true_responses(self.alphas[start_items], self.intercepts[start_items], start_items,
                                           response_vec=true_responses.astype(int))
        else:
            self.tts.answer_items(self.alphas[start_items], self.intercepts[start_items], start_items)
        self.tts.get_pos_mean(0, self.mc_samples)
        for j in tqdm(range(1, self.max_items)):
            new_items = self.select_items(j)
            if self.plugin_response:
                true_responses = self.response[np.arange(self.response.shape[0]), new_items]
                self.tts.answer_true_responses(self.alphas[new_items], self.intercepts[new_items], new_items,
                                               response_vec=true_responses.astype(int))
            else:
                self.tts.answer_items(self.alphas[new_items], self.intercepts[new_items], new_items)
            self.tts.get_pos_mean(j, self.mc_samples)
        self.theta_updates = self.tts.pos_mean
        self.item_selections = self.tts.item_ids

    def simulate_mi_multistep_look_ahead(self, H=1, gamma=0.99):
        """For each test taker, perfom h-step look-ahead MI selections"""
        # start_items = np.array([self.start]*self.n)
        # if self.plugin_response:
        #     true_responses = self.response[np.arange(self.response.shape[0]), start_items]
        #     self.tts.answer_true_responses(self.alphas[start_items], self.intercepts[start_items], start_items,
        #                                    response_vec=true_responses.astype(int))
        # else:
        #     self.tts.answer_items(self.alphas[start_items], self.intercepts[start_items], start_items)
        for j in tqdm(range(self.max_items)):
            new_items = self.select_items_mi_bellman(j, H, gamma)
            if self.plugin_response:
                true_responses = self.response[np.arange(self.response.shape[0]), new_items]
                self.tts.answer_true_responses(self.alphas[new_items], self.intercepts[new_items], new_items,
                                               response_vec=true_responses.astype(int))
            else:
                self.tts.answer_items(self.alphas[new_items], self.intercepts[new_items], new_items)
        self.theta_updates = self.tts.pos_mean
        self.item_selections = self.tts.item_ids

    # def simulate_mi_multistep_look_ahead(self, H=1, gamma=0.99):
    #     """For each test taker, perfom h-step look-ahead MI selections"""
    #     start_items = np.array([self.start]*self.n)
    #     if self.plugin_response:
    #         true_responses = self.response[np.arange(self.response.shape[0]), start_items]
    #         self.tts.answer_true_responses(self.alphas[start_items], self.intercepts[start_items], start_items,
    #                                        response_vec=true_responses.astype(int))
    #     else:
    #         self.tts.answer_items(self.alphas[start_items], self.intercepts[start_items], start_items)
    #     for j in tqdm(range(1, self.max_items)):
    #         new_items = self.select_items_mi_bellman(j, H, gamma)
    #         if self.plugin_response:
    #             true_responses = self.response[np.arange(self.response.shape[0]), new_items]
    #             self.tts.answer_true_responses(self.alphas[new_items], self.intercepts[new_items], new_items,
    #                                            response_vec=true_responses.astype(int))
    #         else:
    #             self.tts.answer_items(self.alphas[new_items], self.intercepts[new_items], new_items)
    #     self.theta_updates = self.tts.pos_mean
    #     self.item_selections = self.tts.item_ids

    def compute_bellman_mi(self, pos_samples: np.ndarray, H: int, gamma: float, action: float, r_a: np.array):
        """Recursion function to compute multistep lookahead, MI bellmam"""
        if H == 0:
            return 0
        num_samples = pos_samples.shape[0]
        # get future samples
        prob1 = norm.cdf(pos_samples@self.alphas[action] + self.intercepts[action])  # sir_large_samples
        pred_hat = np.mean(prob1)
        threshold = 1e-10
        prob1 = np.where(prob1 > (1 - threshold), 1 - threshold, prob1)
        prob1 = np.where(prob1 < threshold, threshold, prob1)
        prob0 = 1 - prob1
        weight1 = prob1 / np.sum(prob1)  # j * sir_large_samples
        weight0 = prob0 / np.sum(prob0)
        indices0 = np.random.choice(a=num_samples, size=num_samples, p=weight0)
        indices1 = np.random.choice(a=num_samples, size=num_samples, p=weight1)
        future_pos_1 = pos_samples[indices1]
        future_pos_0 = pos_samples[indices0]
        # compute mi
        p_sampled0 = 1 - norm.cdf(future_pos_0@self.alphas[action]+self.intercepts[action])
        p_sampled1 = norm.cdf(future_pos_1@self.alphas[action]+self.intercepts[action])
        mi_term1 = pred_hat * (np.mean(np.log(p_sampled1)) - np.log(pred_hat))
        mi_term2 = (1 - pred_hat) * (np.mean(np.log(p_sampled0)) - np.log(1-pred_hat))
        mi = mi_term1 + mi_term2
        new_ra = np.setdiff1d(r_a, action)
        one_subtree = np.max([self.compute_bellman_mi(future_pos_1, H-1, gamma, a, new_ra) for a in new_ra])
        zero_subtree = np.max([self.compute_bellman_mi(future_pos_0, H - 1, gamma, a, new_ra) for a in new_ra])
        return mi + gamma * (pred_hat * one_subtree + (1-pred_hat)*zero_subtree)

    def select_items_mi_bellman(self, item_idx:int, H:int, gamma: float):
        """Find items by performing multistep lookahead for MI"""
        result = np.zeros(self.n, dtype=int)
        if item_idx == 0:
            new_samples = np.random.multivariate_normal(np.zeros(self.k),
                                                        np.identity(self.k), size=(self.n, self.sir_large_samples))
        else:
            new_samples = self.tts.get_factor_samples(self.sir_large_samples)  # n by num_sir_samples by k
        for i in range(self.n):
            print(i)
            avail_items = np.setdiff1d(np.arange(self.m), self.tts.item_ids[i, :item_idx])
            cum_rewards = np.zeros(avail_items.shape[0])
            for j, a in enumerate(avail_items):
                cum_rewards[j] = self.compute_bellman_mi(new_samples[i], H, gamma, a, avail_items)
            result[i] = avail_items[np.argmax(cum_rewards)]
        return result

    def select_kl_eap(self, item_idx: int):
        """Select Items Based on Posterior Expected KL Information"""
        result = np.zeros(self.n, dtype=int)
        pos_mean = self.tts.pos_mean[item_idx-1] # n by k
        new_samples = self.tts.get_factor_samples(self.mc_samples) # n by num_samples by k
        for i in range(self.n):
            theta_hat = pos_mean[i].reshape(-1, 1)
            theta_sampled = new_samples[i] # num_samples by k
            avail_items = np.setdiff1d(np.arange(self.m), self.tts.item_ids[i, :item_idx])
            p_hat = norm.cdf(self.alphas[avail_items, :] @ theta_hat.flatten() + self.intercepts[avail_items]) # j-dim vec
            p_sampled = norm.cdf(self.alphas[avail_items, :] @ theta_sampled.T + self.intercepts[avail_items].reshape(-1,1)) # j * mc_samples
            kl_info = p_hat * np.mean(
                np.log(1/p_sampled * p_hat.reshape(-1, 1)), axis=1) + (1-p_hat) * np.mean(
                np.log(1/(1-p_sampled) * (1-p_hat).reshape(-1, 1)), axis=1)
            result[i] = avail_items[np.argmax(kl_info)]
        return result

    def select_kl_pos(self, item_idx: int):
        """Select Items Based on KL Distance Between Subsequent Posteriors"""
        result = np.zeros(self.n, dtype=int)
        all_items = np.arange(self.m)
        pos_pred = self.tts.get_pos_pred(item_idx, self.mc_samples, self.alphas, self.intercepts)  # n by num_avail_items
        new_samples = self.tts.get_factor_samples(self.mc_samples)  # n by num_samples by k
        for i in range(self.n):
            pred_hat = pos_pred[i]
            theta_sampled = new_samples[i]
            avail_items = all_items[~np.in1d(all_items, self.tts.item_ids[i, :item_idx])]
            p_sampled = norm.cdf(self.alphas[avail_items, :] @ theta_sampled.T + self.intercepts[avail_items].reshape(-1,1)) # j * mc_samples
            kl_term1 = pred_hat * np.mean(np.log(1/p_sampled * pred_hat.reshape(-1,1)), axis=1)
            kl_term2 = (1-pred_hat) * np.mean(np.log(1/(1-p_sampled) * (1-pred_hat).reshape(-1,1)), axis=1)
            result[i] = avail_items[np.argmax(kl_term1+kl_term2)]
        return result

    def select_mi(self, item_idx: int):
        """Select Items Based on Mutual Information"""
        result = np.zeros(self.n, dtype=int)
        all_items = np.arange(self.m)
        pos_pred = self.tts.get_pos_pred(item_idx, self.mc_samples, self.alphas, self.intercepts)  # n by num_avail_items
        for i in range(self.n):
            pred_hat = pos_pred[i]
            avail_items = all_items[~np.in1d(all_items, self.tts.item_ids[i, :item_idx])]
            theta_sampled0 = self.tts.get_next_factor_samples(i, 0, self.mc_samples, self.alphas[avail_items,:],
                                                        self.intercepts[avail_items]) # (avial_items, mc_samples, k)
            theta_sampled1 = self.tts.get_next_factor_samples(i, 1, self.mc_samples, self.alphas[avail_items, :],
                                                         self.intercepts[avail_items])  # (avial_items, mc_samples, k)
            p_sampled0 = 1- norm.cdf(
                np.squeeze(np.matmul(theta_sampled0, self.alphas[avail_items, :, np.newaxis]), axis=-1
                           ) +self.intercepts[avail_items].reshape(-1,1)) # j by mc samples
            p_sampled1 = norm.cdf(
                np.squeeze(np.matmul(theta_sampled1, self.alphas[avail_items, :, np.newaxis]), axis=-1
                           ) + self.intercepts[avail_items].reshape(-1, 1))  # j by mc samples
            mi_term1 = pred_hat * np.mean(np.log(p_sampled1 / pred_hat.reshape(-1, 1)), axis=1)
            mi_term2 = (1 - pred_hat) * np.mean(np.log(p_sampled0 / (1 - pred_hat).reshape(-1, 1)), axis=1)
            result[i] = avail_items[np.argmax(mi_term1 + mi_term2)]
        return result

    def select_mi_sir(self, item_idx: int):
        """Select Items Based on Mutual Information, but use posterior reweighting to accelerate"""
        result = np.zeros(self.n, dtype=int)
        all_items = np.arange(self.m)
        pos_pred = self.tts.get_pos_pred(item_idx, self.mc_samples, self.alphas,
                                         self.intercepts)  # n by num_avail_items
        new_samples = self.tts.get_factor_samples(self.sir_large_samples)  # n by num_sir_samples by k
        for i in range(self.n):
            pred_hat = pos_pred[i]
            theta_sampled = new_samples[i]
            avail_items = all_items[~np.in1d(all_items, self.tts.item_ids[i, :item_idx])]
            reweighted_samples = self.tts.get_sir_reweighted_samples(theta_sampled, self.sir_samples,
                                                                     self.alphas[avail_items, :],
                                                                     self.intercepts[avail_items])
            p_sampled0 = 1 - norm.cdf(
                np.squeeze(np.matmul(reweighted_samples["0"], self.alphas[avail_items, :, np.newaxis]), axis=-1
                           ) + self.intercepts[avail_items].reshape(-1, 1))  # j by mc samples
            p_sampled1 = norm.cdf(
                np.squeeze(np.matmul(reweighted_samples["1"], self.alphas[avail_items, :, np.newaxis]), axis=-1
                           ) + self.intercepts[avail_items].reshape(-1, 1))  # j by mc samples
            mi_term1 = pred_hat * np.mean(np.log(p_sampled1 / pred_hat.reshape(-1, 1)), axis=1)
            mi_term2 = (1 - pred_hat) * np.mean(np.log(p_sampled0 / (1 - pred_hat).reshape(-1, 1)), axis=1)
            result[i] = avail_items[np.argmax(mi_term1 + mi_term2)]
        return result


    def select_kl_oracle(self, item_idx: int):
        """Currently Assume Each Person Only have one single oracle distribution (regardless what they get for
        the next step item) """
        result = np.zeros(self.n, dtype=int)
        all_items = np.arange(self.m)
        pos_pred = self.tts.get_pos_pred(item_idx, self.mc_samples, self.alphas, self.intercepts)
        num_items = pos_pred.shape[1]
        U_hat = (self.rng.uniform(0, 1, pos_pred.shape) < pos_pred).astype(int) # n by num_avarialbe items
        for i in range(self.n):
            if i % 50 == 0:
                print("person", i)
            pred_hat = pos_pred[i]
            avail_items = all_items[~np.in1d(all_items, self.tts.item_ids[i, :item_idx])]
            theta_next0 = self.tts.get_next_factor_samples(i, 0, self.mc_samples, self.alphas[avail_items, :],
                                                              self.intercepts[avail_items])  # (avial_items, mc_samples, k)
            theta_next1 = self.tts.get_next_factor_samples(i, 1, self.mc_samples, self.alphas[avail_items, :],
                                                              self.intercepts[avail_items])  # (avial_items, mc_samples, k)
            oracle_thetas = self.tts.get_exact_oracle_samples(i, self.mc_samples, self.alphas[avail_items, :],
                                                              self.intercepts[avail_items], U_hat[i])

            # compute the log denumerator for each item
            log_pred0, log_pred1 = np.zeros(num_items), np.zeros(num_items)
            threshold = 1e-10
            for j in range(num_items):
                # poster when u_{jm} = 0
                temp0 = norm.cdf(theta_next0[j] @ self.alphas[avail_items, :].T + self.intercepts[avail_items].reshape(1,-1))
                temp0 = np.where(temp0 > (1 - threshold), 1 - threshold, temp0)
                temp0 = np.where(temp0 < threshold, threshold,temp0)
                temp0_comp = 1 - temp0 # s by j
                log_pred0_final = np.empty_like(temp0)
                log_pred0_final[:,U_hat[i]==0] = temp0_comp[:, U_hat[i]==0]
                log_pred0_final[:, U_hat[i] == 1] = temp0[:, U_hat[i] == 1]
                log_pred0_final = np.log(log_pred0_final)
                log_pred0[j] = np.mean(np.sum(log_pred0_final, axis=1) - log_pred0_final[:, j])
                # poster when u_{jm} = 1
                temp1 = norm.cdf(
                    theta_next0[1] @ self.alphas[avail_items, :].T + self.intercepts[avail_items].reshape(1, -1))
                temp1= np.where(temp1 > (1 - threshold), 1 - threshold, temp1)
                temp1 = np.where(temp1 < threshold, threshold, temp1)
                temp1_comp = 1 - temp1  # s by j
                log_pred1_final = np.empty_like(temp0)
                log_pred1_final[:, U_hat[i] == 0] = temp1_comp[:, U_hat[i] == 0]
                log_pred1_final[:, U_hat[i] == 1] = temp1[:, U_hat[i] == 1]
                log_pred1_final = np.log(log_pred1_final)
                log_pred1[j] = np.mean(np.sum(log_pred1_final, axis=1) - log_pred1_final[:, j])

            # compute the log numerator for each item
            p_mat = norm.cdf(oracle_thetas @ self.alphas[avail_items, :].T + self.intercepts[avail_items].reshape(1, -1))
            p_mat = np.where(p_mat > (1 - threshold), 1 - threshold, p_mat)
            p_mat = np.where(p_mat < threshold, threshold, p_mat)
            p_mat = p_mat.T # J by S
            p_mat_comp = 1 - p_mat
            p_mat_final = np.empty_like(p_mat)
            p_mat_final[U_hat[i] == 0] = p_mat_comp[U_hat[i] == 0]
            p_mat_final[U_hat[i] == 1] = p_mat[U_hat[i] == 1]
            p_mat_final = np.log(p_mat_final) # j by s
            log_p_vec = np.zeros(num_items)
            for j in range(num_items):
                log_p_vec[j] = np.mean(np.sum(np.delete(p_mat_final, j, axis=0), axis=0))

            kl_term1 = pred_hat * (log_p_vec - log_pred1)
            kl_term2 = (1 - pred_hat) * (log_p_vec - log_pred0)
            result[i] = avail_items[np.argmin(kl_term1 + kl_term2)]
        return result

    def select_var_e_pred(self, item_idx):
        """Select Items with the largest Predictive varainces around its mean int (p-c)^2 """
        result = np.zeros(self.n, dtype=int)
        all_items = np.arange(self.m)
        new_samples = self.tts.get_factor_samples(self.mc_samples)  # n by num_sir_samples by k
        for i in range(self.n):
            theta_sampled = new_samples[i] # s by k
            avail_items = all_items[~np.in1d(all_items, self.tts.item_ids[i, :item_idx])]
            linear_term = theta_sampled @ self.alphas[avail_items].T + self.intercepts[avail_items]
            pred = norm.cdf(linear_term) # s by j
            pred_var = np.var(pred, axis = 0)
            result[i] = avail_items[np.argmax(pred_var)]
        return result

    def select_e_var_pred(self, item_idx):
        """Select Items with the largest Predictive varainces integral over p(1-p)

        Note: select_pred_var is the variance of prediction mean
        """
        result = np.zeros(self.n, dtype=int)
        all_items = np.arange(self.m)
        new_samples = self.tts.get_factor_samples(self.mc_samples)  # n by num_sir_samples by k
        for i in range(self.n):
            theta_sampled = new_samples[i] # s by k
            avail_items = all_items[~np.in1d(all_items, self.tts.item_ids[i, :item_idx])]
            linear_term = theta_sampled @ self.alphas[avail_items].T + self.intercepts[avail_items]
            pred = norm.cdf(linear_term) # s by j
            pred_var = np.mean(pred*(1-pred), axis = 0)
            result[i] = avail_items[np.argmax(pred_var)]
        return result

    def select_true_var_pred(self, item_idx):
        """Select Items with the largest Predictive varainces integral over p(1-p) + another integral over (p-c)^2

        Note: var(y|d) = E[var(y|\theta)] + var[E[y|theta]]
        """
        result = np.zeros(self.n, dtype=int)
        all_items = np.arange(self.m)
        new_samples = self.tts.get_factor_samples(self.mc_samples)  # n by num_sir_samples by k
        for i in range(self.n):
            theta_sampled = new_samples[i]  # s by k
            avail_items = all_items[~np.in1d(all_items, self.tts.item_ids[i, :item_idx])]
            linear_term = theta_sampled @ self.alphas[avail_items].T + self.intercepts[avail_items]
            pred = norm.cdf(linear_term)  # s by j
            e_var_pred = np.mean(pred * (1 - pred), axis=0)
            var_e_pred = np.var(pred, axis = 0)
            true_var = e_var_pred + var_e_pred
            result[i] = avail_items[np.argmax(true_var)]
        return result

    def select_interquartile_range(self, item_idx):
        """Select Items with the largest interquartile range, in terms of the distribution of prediction mean """
        result = np.zeros(self.n, dtype=int)
        all_items = np.arange(self.m)
        new_samples = self.tts.get_factor_samples(self.mc_samples)  # n by num_sir_samples by k
        for i in range(self.n):
            theta_sampled = new_samples[i] # s by k
            avail_items = all_items[~np.in1d(all_items, self.tts.item_ids[i, :item_idx])]
            linear_term = theta_sampled @ self.alphas[avail_items].T + self.intercepts[avail_items]
            pred = norm.cdf(linear_term) # s by j
            quartile_3rd = np.percentile(pred, q=0.75, axis=0)
            quartile_1st= np.percentile(pred, q=0.25, axis=0)
            ir = quartile_3rd-quartile_1st
            result[i] = avail_items[np.argmax(ir)]
        return result

    def select_pred_mean(self, item_idx):
        """Select Items with predictive mean closest to 0.5 """
        result = np.zeros(self.n, dtype=int)
        all_items = np.arange(self.m)
        new_samples = self.tts.get_factor_samples(self.mc_samples)  # n by num_sir_samples by k
        for i in range(self.n):
            theta_sampled = new_samples[i]  # s by k
            avail_items = all_items[~np.in1d(all_items, self.tts.item_ids[i, :item_idx])]
            linear_term = theta_sampled @ self.alphas[avail_items].T + self.intercepts[avail_items]
            pred = norm.cdf(linear_term)  # s by j
            pred_mean = np.abs(np.mean(pred, axis=0) - 0.5)
            result[i] = avail_items[np.argmin(pred_mean)]
        return result

    def select_pred_entropy(self, item_idx):
        """Select Items with the largest predictive entropy"""
        result = np.zeros(self.n, dtype=int)
        all_items = np.arange(self.m)
        new_samples = self.tts.get_factor_samples(self.mc_samples)  # n by num_sir_samples by k
        for i in range(self.n):
            theta_sampled = new_samples[i]  # s by k
            avail_items = all_items[~np.in1d(all_items, self.tts.item_ids[i, :item_idx])]
            linear_term = theta_sampled @ self.alphas[avail_items].T + self.intercepts[avail_items]
            pred = norm.cdf(linear_term)  # s by j
            pred0 = 1-pred
            log_pred = np.log(pred)
            log_pred0 = np.log(1-pred)
            neg_entropy = np.mean(pred*log_pred + pred0*log_pred0, axis=0)
            result[i] = avail_items[np.argmin(neg_entropy)]
        return result

    def select_jsd(self, item_idx):
        """ Replace KL with Jensen-Shannon Divergence

        One can show Mutual information is the expected Kl divergence between predicted mean weighted by posterior and
        and the constant posterior predicted mean (see update1). What if we replace KL with above
        """
        result = np.zeros(self.n, dtype=int)
        all_items = np.arange(self.m)
        pos_pred = self.tts.get_pos_pred(item_idx, self.mc_samples, self.alphas,
                                         self.intercepts)  # n by num_avail_items
        new_samples = self.tts.get_factor_samples(self.mc_samples)  # n by num_sir_samples by k
        for i in range(self.n):
            theta_sampled = new_samples[i] # s by k
            avail_items = all_items[~np.in1d(all_items, self.tts.item_ids[i, :item_idx])]
            linear_term = theta_sampled @ self.alphas[avail_items].T + self.intercepts[avail_items]
            pred = norm.cdf(linear_term) # s by j
            pred0 = 1-pred
            c = pos_pred[i] # j dimensional
            term1 = pred * np.log(2*pred/(pred+c)) # s by j
            term2 = pred0 * np.log(2*pred0/(1+pred0-c))
            term3 = c* np.log((2*c)/(pred+c))
            term4 = (1-c)* np.log((2-2*c)/(1+pred0-c))
            jsd = np.mean(term1+term2+term3+term4, axis=0) # ignore 1/2
            result[i] = avail_items[np.argmax(jsd)]
        return result

    def select_hellinger(self, item_idx):
        """ Replace KL with hellinger divergence

        One can show Mutual information is the expected Kl divergence between predicted mean weighted by posterior and
        and the constant posterior predicted mean (see update1). What if we replace KL with above
        """
        result = np.zeros(self.n, dtype=int)
        all_items = np.arange(self.m)
        pos_pred = self.tts.get_pos_pred(item_idx, self.mc_samples, self.alphas,
                                         self.intercepts)  # n by num_avail_items
        new_samples = self.tts.get_factor_samples(self.mc_samples)  # n by num_sir_samples by k
        for i in range(self.n):
            theta_sampled = new_samples[i] # s by k
            avail_items = all_items[~np.in1d(all_items, self.tts.item_ids[i, :item_idx])]
            linear_term = theta_sampled @ self.alphas[avail_items].T + self.intercepts[avail_items]
            pred = norm.cdf(linear_term) # s by j
            pred0 = 1-pred
            c = pos_pred[i]  # j dimensional
            sqrtpc = np.sqrt(pred*c)
            sqrtpc0 = np.sqrt(pred0*(1-c))
            squared_hellinger = np.mean(1-sqrtpc - sqrtpc0, axis=0)
            result[i] = avail_items[np.argmax(squared_hellinger)]
        return result

    def select_var_reduction(self, item_idx):
        """Select next items to reduce posterior variance"""
        result = np.zeros(self.n, dtype=int)
        all_items = np.arange(self.m)
        pos_pred = self.tts.get_pos_pred(item_idx, self.mc_samples, self.alphas,
                                         self.intercepts)  # n by num_avail_items
        new_samples = self.tts.get_factor_samples(self.sir_large_samples)  # n by num_sir_samples by k
        for i in range(self.n):
            pred_hat = pos_pred[i]
            theta_sampled = new_samples[i]
            avail_items = all_items[~np.in1d(all_items, self.tts.item_ids[i, :item_idx])]
            reweighted_samples = self.tts.get_sir_reweighted_samples(theta_sampled, self.mc_samples,
                                                                     self.alphas[avail_items, :],
                                                                     self.intercepts[avail_items]) # (j by m by k)
            mean = pred_hat.reshape(-1,1) * np.mean(reweighted_samples["1"], axis=1) + (1-pred_hat.reshape(-1,1)) * np.mean(reweighted_samples["0"], axis=1) # j by k
            second_moments = pred_hat.reshape(-1,1) * np.mean(reweighted_samples["1"]**2, axis=1) + (1-pred_hat.reshape(-1,1)) * np.mean(reweighted_samples["0"]**2, axis=1)
            variance = np.sum(second_moments-mean**2, axis=1)
            result[i] = avail_items[np.argmax(variance)]
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
        elif self.sc == "kl_oracle": # kl oracle
            return self.select_kl_oracle(item_idx)
        elif self.sc == "predictive_e_variance":
            return self.select_e_var_pred(item_idx)
        elif self.sc == "predictive_variance_e":
            return self.select_var_e_pred(item_idx)
        elif self.sc == "true_predictive_variance":
            return self.select_true_var_pred(item_idx)
        elif self.sc == "predictive_mean":
            return self.select_pred_mean(item_idx)
        elif self.sc == "predictive_entropy":
            return self.select_pred_entropy(item_idx)
        elif self.sc == "variance_reduction":
            return self.select_var_reduction(item_idx)
        elif self.sc == "jsd":
            return self.select_jsd(item_idx)
        elif self.sc == "hellinger":
            return self.select_hellinger(item_idx)
        elif self.sc == "interquartile_range":
            return self.select_interquartile_range(item_idx)
        else:
            raise ValueError("unknown selection criterion")

    def factor_inference(self, num_samples: int, ci: float =0.95):
        """Get Posterior Means and Confidence Intervals for the Final Estimates"""
        samples = self.tts.get_factor_samples(num_samples) # (n, num_samples, k)
        pos_mean = np.mean(samples, axis=1) # n by k
        ci_lb = np.quantile(samples, (1-ci)/2, axis=1)
        ci_ub = np.quantile(samples, ci + (1-ci)/2, axis=1)
        return {"pos_mean": pos_mean, "ci_lb": ci_lb, "ci_ub": ci_ub, "samples": samples}

















