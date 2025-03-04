import typing
import numpy as np
from .test_taker import TestTakers
from tqdm import tqdm
from scipy.stats import norm


class OptimalPolicyCAT:

    def __init__(
            self,
            true_thetas: np.ndarray,
            test_bank: typing.Dict[str, np.ndarray],
            selection_criterion: str,
            max_items: int,
            mc_samples: int,
            sir_large_samples: int,
            oracle_samples: np.ndarray,
            starting_item: typing.Optional[int] = None,
            num_workers: int = 8,
            random_state: int = 42,
    ):
        """
        How much better can we do if oracle is known?

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
        self.sir_large_samples = sir_large_samples
        self.num_workers = num_workers
        self.rng = np.random.default_rng(random_state)
        self.m = self.alphas.shape[0]
        self.theta_updates = np.zeros((self.max_items, self.n, self.k))
        self.item_selections = np.zeros((self.n, self.max_items))
        self.oracle_samples = oracle_samples["samples"]
        self.oracle_predictions = self.get_oracle_predictions()
        if starting_item:
            self.start = starting_item
        else:
            self.start = self.pick_starting_item()
        self.sc = selection_criterion
        if self.sc == "kl":
            self.oracle_response = oracle_samples["response"]
            self.oracle_loglikhood = self.get_oracle_loglikhood()
        self.tts = TestTakers(thetas=self.true_thetas, max_item=self.max_items, num_workers=self.num_workers,
                              random_state=self.seed)

    def get_oracle_predictions(self):
        """Return n by m prediction means"""
        preds = np.zeros((self.n, self.m))
        for i in range(self.n):
            samples_i = self.oracle_samples[i].T  # k by num_samples
            preds[i] = np.mean(norm.cdf(self.alphas @ samples_i + self.intercepts.reshape(-1, 1)), axis=1)
        return preds

    def get_oracle_loglikhood(self):
        """For each i, precompute the loglikelihood for each entry - yielding the mc_samples by m matrix

        Then avarage through mc samples and get 1 by m average log predictions!
        """
        result = np.zeros((self.n, self.m))
        threshold = 1e-10
        for i in range(self.n):
            samples_i = self.oracle_samples[i].T  # k by num_samples
            pred1 = norm.cdf(self.alphas @ samples_i + self.intercepts.reshape(-1, 1)) # j by s
            pred1 = np.where(pred1 > (1 - threshold), 1 - threshold, pred1)
            pred1 = np.where(pred1 < threshold, threshold, pred1)
            pred0 = 1 - pred1  # j by s
            response_i = self.oracle_response[i]
            pred_final = np.empty_like(pred1)
            pred_final[response_i==0, :] = pred0[response_i==0, :]
            pred_final[response_i==1, :] = pred0[response_i==1, :]
            log_pred = np.log(pred_final) # j by s
            result[i] = np.mean(log_pred, axis=1)
        return result


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
        self.tts.answer_items(self.alphas[start_items], self.intercepts[start_items], start_items)
        self.tts.get_pos_mean(0, self.mc_samples)
        for j in tqdm(range(1, self.max_items)):
            new_items = self.select_items(j)
            self.tts.answer_items(self.alphas[new_items], self.intercepts[new_items], new_items)
            self.tts.get_pos_mean(j, self.mc_samples)
        self.theta_updates = self.tts.pos_mean
        self.item_selections = self.tts.item_ids

    def select_mahalanobis(self, item_idx: int):
        """Select Items Based on mahalanobis distance, but use posterior reweighting to accelerate"""
        result = np.zeros(self.n, dtype=int)
        all_items = np.arange(self.m)
        new_samples = self.tts.get_factor_samples(self.sir_large_samples)  # n by num_sir_samples by k
        for i in range(self.n):
            theta_sampled = new_samples[i]
            avail_items = all_items[~np.in1d(all_items, self.tts.item_ids[i, :item_idx])]
            # get current posterior statistics
            for q in [0.25,0.5,0.75]:
                self.tts.dist_stats[q][i, item_idx] = np.quantile(new_samples[i], q, axis=0)
            self.tts.dist_stats["mean"][i, item_idx] = np.mean(new_samples[i], axis=0)
            self.tts.dist_stats["cov"][i, item_idx] = np.cov(new_samples[i].T).flatten()
            # select next items
            pred_hat = self.oracle_predictions[i][avail_items]
            reweighted_samples = self.tts.get_sir_reweighted_samples(theta_sampled, self.mc_samples,
                                                                     self.alphas[avail_items, :],
                                                                     self.intercepts[avail_items])
            oracle_temp = self.oracle_samples[i]
            oracle_cov = np.cov(oracle_temp.T)
            oracle_mean = np.mean(oracle_temp, axis = 0)
            inv_oracle_cov =  np.linalg.inv(oracle_cov)
            # zero case
            mean_zero = np.array([np.mean(reweighted_samples["0"][j], axis=0) for j in range(len(avail_items))])
            distance_zero = np.array([np.sqrt(np.dot(np.dot((mean_zero[j] - oracle_mean).T, inv_oracle_cov), (mean_zero[j] - oracle_mean))) for j in range(len(avail_items))])
            # one case
            mean_one = np.array([np.mean(reweighted_samples["0"][1], axis=0) for j in range(len(avail_items))])
            distance_one = np.array(
                [np.sqrt(np.dot(np.dot((mean_one[j] - oracle_mean).T, inv_oracle_cov), (mean_one[j] - oracle_mean)))
                 for j in range(len(avail_items))])
            result[i] = avail_items[np.argmin(distance_one*pred_hat + distance_zero*(1-pred_hat))]
        return result

    def select_kl(self, item_idx):
        """Select based on KL"""
        result = np.zeros(self.n, dtype=int)
        all_items = np.arange(self.m)
        new_samples = self.tts.get_factor_samples(self.sir_large_samples)  # n by num_sir_samples by k
        for i in range(self.n):
            if i % 500 ==0:
                print(i)
            theta_sampled = new_samples[i]
            avail_items = all_items[~np.in1d(all_items, self.tts.item_ids[i, :item_idx])]
            # get current posterior statistics
            for q in [0.25, 0.5, 0.75]:
                self.tts.dist_stats[q][i, item_idx] = np.quantile(new_samples[i], q, axis=0)
            self.tts.dist_stats["mean"][i, item_idx] = np.mean(new_samples[i], axis=0)
            self.tts.dist_stats["cov"][i, item_idx] = np.cov(new_samples[i].T).flatten()
            pred_hat = self.oracle_predictions[i][avail_items]
            reweighted_samples = self.tts.get_sir_reweighted_samples(theta_sampled, self.mc_samples,
                                                                     self.alphas[avail_items, :],
                                                                     self.intercepts[avail_items])
            # compute kl numerator
            sum_logliklihood = np.sum(self.oracle_loglikhood[i, avail_items])
            kl_numerator = np.array([sum_logliklihood- self.oracle_loglikhood[i, item] for item in avail_items])
            # compute kl denumerator part
            num_items = len(avail_items)
            log_pred0, log_pred1 = np.zeros(num_items), np.zeros(num_items)
            threshold = 1e-10
            oracle_response_i = self.oracle_response[i, avail_items]
            for j in range(num_items):
                # poster when u_{jm} = 0
                if oracle_response_i[j] == 0:
                    temp0 = norm.cdf(
                        reweighted_samples["0"][j] @ self.alphas[avail_items, :].T + self.intercepts[avail_items].reshape(1, -1)) # s by j
                    temp0 = np.where(temp0 > (1 - threshold), 1 - threshold, temp0)
                    temp0 = np.where(temp0 < threshold, threshold, temp0)
                    temp0_comp = 1 - temp0  # s by j
                    log_pred0_final = np.empty_like(temp0)
                    log_pred0_final[:, oracle_response_i == 0] = temp0_comp[:, oracle_response_i == 0]
                    log_pred0_final[:, oracle_response_i == 1] = temp0[:, oracle_response_i== 1]
                    log_pred0_final = np.log(log_pred0_final)
                    log_pred0[j] = np.mean(np.sum(log_pred0_final, axis=1) - log_pred0_final[:, j])
                else:
                     # poster when u_{jm} = 1
                    temp1 = norm.cdf(
                        reweighted_samples["1"][j] @ self.alphas[avail_items, :].T + self.intercepts[avail_items].reshape(1, -1))
                    temp1 = np.where(temp1 > (1 - threshold), 1 - threshold, temp1)
                    temp1 = np.where(temp1 < threshold, threshold, temp1)
                    temp1_comp = 1 - temp1  # s by j
                    log_pred1_final = np.empty_like(temp1)
                    log_pred1_final[:, oracle_response_i == 0] = temp1_comp[:, oracle_response_i == 0]
                    log_pred1_final[:, oracle_response_i == 1] = temp1[:, oracle_response_i == 1]
                    log_pred1_final = np.log(log_pred1_final)
                    log_pred1[j] = np.mean(np.sum(log_pred1_final, axis=1) - log_pred1_final[:, j])
            result[i] = avail_items[np.argmin(kl_numerator - log_pred1 - log_pred0)]
        return result


    def select_items(self, item_idx):
        """Select Next Items Depending on Given Selection Criterion for each test taker"""
        if self.sc == "mahalanobis":
            return self.select_mahalanobis(item_idx)
        elif self.sc == "kl":
            return self.select_kl(item_idx)

    def factor_inference(self, num_samples: int, ci: float =0.95):
        """Get Posterior Means and Confidence Intervals for the Final Estimates"""
        samples = self.tts.get_factor_samples(num_samples) # (n, num_samples, k)
        pos_mean = np.mean(samples, axis=1) # n by k
        ci_lb = np.quantile(samples, (1-ci)/2, axis=1)
        ci_ub = np.quantile(samples, ci + (1-ci)/2, axis=1)
        return {"pos_mean": pos_mean, "ci_lb": ci_lb, "ci_ub": ci_ub, "samples": samples}







# @ staticmethod
    # def compute_kde_1d(new_samples, reference_samples, bandwidth=0.5):
    #     kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    #     kde.fit(new_samples[:, np.newaxis])
    #     return np.exp(kde.score_samples(reference_samples[: , np.newaxis]))
    #
    # def compute_oracle_marginal_kl(self):
    #     result = np.zeros(self.oracle_samples.shape)
    #     for i in range(self.n):
    #         for k in range(self.k):
    #             result[i][:, k] = self.compute_kde_1d(self.oracle_samples[i][:, k], self.oracle_samples[i][:, k])
    #     return result






