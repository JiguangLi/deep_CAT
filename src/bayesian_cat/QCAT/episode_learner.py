import numpy as np
from numpy.linalg import inv
import scipy.stats as stats
from scipy.stats import norm
from .minimax_tilting import TruncatedMVN


class EpisodeLearner:

    def __init__(
            self,
            point_theta: np.array,
            num_items : int,
            max_item: int,
            num_workers: int,
            eval_samples: int,
            random_state: int = 42
    ):
        """
        Create Test Taker Object with true ability theta

            Args:
                thetas: np.array
                num_items: num items in test bank
                max_item: maximum amount of items administered to this test taker
                num_workers: number of cores for parallel computing
                random_state: seed
        """
        self.rng = np.random.default_rng(random_state)
        self.random_state = random_state
        self.num_workers = num_workers
        self.thetas = point_theta
        self.k = point_theta.shape[0]
        self.m = num_items
        max_item = max_item + 1 # since we filled index zero with zeros
        self.max_item = max_item
        self.eval_samples =eval_samples
        self.d1_array = np.zeros((max_item, self.k))
        self.d2_array = np.zeros(max_item)
        self.s_array = np.zeros(max_item)
        self.i_count = 0
        self.item_ids = np.ones(max_item)*(-1)
        self.item_ids = self.item_ids.astype(int)
        self.item_responses = np.zeros(max_item)
        cov_params = int((1+self.k)*self.k/2)
        self.dist_stats = {0.25: np.zeros((max_item, self.k)),
                           0.5: np.zeros((max_item, self.k)),
                           0.75: np.zeros((max_item, self.k)),
                           "mean": np.zeros((max_item, self.k)),
                           "cov": np.zeros((max_item, cov_params))}
        self.dist_stats[0.25][0, :] = -0.67448
        self.dist_stats[0.5][0, :] = 0
        self.dist_stats[0.75][0, :] = 0.67448
        cov_mat = np.identity(self.k)
        self.dist_stats["cov"][0, :] = cov_mat[np.triu_indices_from(cov_mat)]

    def get_state(self, time_horizon):
        """Get state, note one should run this only if the running statistics has been computed!!!

        e.g. run after eval_latent_traits()
        """
        if time_horizon > self.i_count:
            raise ValueError("Can't query the future states")
        else:
            x = np.concatenate([np.concatenate((self.d1_array[j],
                                                np.array([self.d2_array[j], self.s_array[j]]),
                                                )) for j in range(time_horizon+1)])
            x_stats = np.concatenate((self.dist_stats[0.25][time_horizon],
                                      self.dist_stats[0.5][time_horizon],
                                      self.dist_stats[0.75][time_horizon],
                                      self.dist_stats["mean"][time_horizon],
                                      self.dist_stats["cov"][time_horizon]))
            mask = np.ones(self.m)
            mask[self.item_ids[:time_horizon].astype(int)] = 0
            state = (x, x_stats, mask)
            return state

    def answer_item(self, alpha: np.array, intercept: float, new_items: int):
        """Simulate the process of answering an item given its loading and intercept
            Args:
                alphas: n by k dimensional array
                intercepts: float
                new_items: item index
        """
        self.i_count += 1
        prob = norm.cdf(np.dot(alpha, self.thetas) + intercept)
        response = (np.random.uniform(0, 1) < prob).astype(int)
        self.d1_array[self.i_count] = (2*response-1)* alpha
        self.d2_array[self.i_count] = (2*response-1)* intercept
        self.s_array[self.i_count] = (np.sum(alpha**2) + 1)**0.5
        self.item_ids[self.i_count-1] = new_items
        self.item_responses[self.i_count-1] = response

    def eval_latent_traits(self, threshold: float):
        """Inference on Latent Traits"""
        i_ub = self.i_count + 1
        samples = self.sample_from_sun(self.d1_array[1:i_ub],
                                       self.s_array[1:i_ub],
                                       self.d2_array[1:i_ub],
                                       self.eval_samples) # s by k
        for q in [0.25, 0.5, 0.75]:
            self.dist_stats[q][self.i_count] = np.quantile(samples, q, axis=0)
        self.dist_stats["mean"][self.i_count] = np.mean(samples, axis=0)
        cov_mat = np.cov(samples.T)
        self.dist_stats["cov"][self.i_count] = cov_mat[np.triu_indices_from(cov_mat)]
        if np.max(np.diag(cov_mat)) > threshold:
            reward = -1
            done = False
        else:
            reward = 0
            done = True
        return reward, done

    def sample_from_sun(self, d1i, s_array, d2_array, num_samples):
        """Sample from the Unified Skew Normal Distributions"""
        item_count = d1i.shape[0]
        cov_v0 = np.identity(self.k) - d1i.T @ (inv(d1i @ d1i.T + np.identity(item_count))) @ d1i
        cov_v1 = np.diag(1 / s_array) @ (d1i @ d1i.T + np.identity(item_count)) @ np.diag(1 / s_array)
        truc_lev = - np.diag(1 / s_array) @ (d2_array.reshape(-1, 1)).flatten()
        v0_s = self.rng.multivariate_normal(np.zeros(self.k), cov_v0, num_samples)
        # note if we set seed=self.random_state, then it will return the same thing for a given params!
        if cov_v1.shape[0] == 1: # we use scipy sampling if the dimension of truncated normal is 1
            v1_s = stats.truncnorm.rvs(truc_lev, np.inf, loc=0, scale=cov_v1, size=(1, num_samples))
        else:
            v1_s = TruncatedMVN(np.zeros(item_count), cov_v1, truc_lev,
                                np.ones_like(truc_lev) * np.inf, seed=None).sample(num_samples)
        linear_t = d1i.T @ (inv(d1i @ d1i.T + np.identity(item_count))) @ np.diag(s_array)
        return v0_s + (linear_t @ v1_s).T


###############################################################################################

class OnlineEpisodeLearner:
    """Episode Learner + Online predictions"""
    def __init__(
            self,
            point_theta: np.array,
            alphas: np.ndarray,
            intercepts: np.ndarray,
            num_items : int,
            max_item: int,
            reward_type: str,
            pred_type: str,
            num_workers: int,
            eval_samples: int,
            random_state: int = 42
    ):
        """
        Create Test Taker Object with true ability theta

            Args:
                thetas: np.array
                num_items: num items in test bank
                max_item: maximum amount of items administered to this test taker
                pred_type: "basic": variance and mean , "detailed": all percentiles
                num_workers: number of cores for parallel computing
                random_state: seed
        """
        self.rng = np.random.default_rng(random_state)
        self.random_state = random_state
        self.reward_type = reward_type
        self.num_workers = num_workers
        self.thetas = point_theta
        self.alphas = alphas
        self.intercepts = intercepts
        self.k = point_theta.shape[0]
        self.m = num_items
        max_item = max_item + 1 # since we filled index zero with zeros
        self.max_item = max_item
        self.eval_samples =eval_samples
        self.d1_array = np.zeros((max_item, self.k))
        self.d2_array = np.zeros(max_item)
        self.s_array = np.zeros(max_item)
        self.i_count = 0
        self.item_ids = np.ones(max_item)*(-1)
        self.item_ids = self.item_ids.astype(int)
        self.item_responses = np.zeros(max_item)
        cov_params = int((1+self.k)*self.k/2)
        self.dist_stats = {0.25: np.zeros((max_item, self.k)),
                           0.5: np.zeros((max_item, self.k)),
                           0.75: np.zeros((max_item, self.k)),
                           "mean": np.zeros((max_item, self.k)),
                           "cov": np.zeros((max_item, cov_params))}
        self.dist_stats[0.25][0, :] = -0.67448
        self.dist_stats[0.5][0, :] = 0
        self.dist_stats[0.75][0, :] = 0.67448
        cov_mat = np.identity(self.k)
        self.dist_stats["cov"][0, :] = cov_mat[np.triu_indices_from(cov_mat)]
        self.pred_type = pred_type
        if pred_type == "basic":
            self.preds = np.zeros((max_item, self.m*2))
        else:
            self.preds = np.zeros((self.m, 11)) # mean variance: 10, 20, ...,90, actually no need to store it for all
        self.initialize_predictions()

    def initialize_predictions(self):
        """Initialize Predictions for Multivariate Normal"""
        thetas = self.rng.multivariate_normal(np.zeros(self.k), np.identity(self.k), self.eval_samples)
        linear_term = thetas @ self.alphas.T + self.intercepts
        pred = norm.cdf(linear_term)  # s by j
        if self.pred_type == "basic":
            pred_mean = np.mean(pred, axis=0)
            pred_var = np.var(pred, axis=0)
            self.preds[0, :self.m] = pred_var
            self.preds[0, -self.m:] = pred_mean
        else:
            self.preds[:, 0] = np.mean(pred, axis=0)
            self.preds[:, 1] = np.var(pred, axis=0)
            for i, q in enumerate(np.arange(0.1, 1, 0.1)):
                self.preds[:, (i+2)] = np.quantile(pred, q=q, axis=0)

    def get_state(self, time_horizon, preds_only=False):
        """Get state, note one should run this only if the running statistics has been computed!!!

        e.g. run after eval_latent_traits()
        """
        if time_horizon > self.i_count:
            raise ValueError("Can't query the future states")
        else:
            if not preds_only:
                x = np.concatenate([np.concatenate((self.d1_array[j],
                                                    np.array([self.d2_array[j], self.s_array[j]]),
                                                    )) for j in range(time_horizon+1)])
                x_stats = np.concatenate((self.dist_stats[0.25][time_horizon],
                                          self.dist_stats[0.5][time_horizon],
                                          self.dist_stats[0.75][time_horizon],
                                          self.dist_stats["mean"][time_horizon],
                                          self.dist_stats["cov"][time_horizon]))
            if self.pred_type == "basic":
                x_preds = self.preds[time_horizon]
                x_preds[self.item_ids[:time_horizon].astype(int)] = 0 # make sure prediction variance for selected items is zero
            else:
                x_preds = self.preds
            mask = np.ones(self.m)
            mask[self.item_ids[:time_horizon].astype(int)] = 0
            if not preds_only:
                state = (x, x_stats, x_preds, mask)
            else:
                state = (x_preds, mask)
            return state

    def answer_item(self, alpha: np.array, intercept: float, new_items: int):
        """Simulate the process of answering an item given its loading and intercept
            Args:
                alphas: n by k dimensional array
                intercepts: float
                new_items: item index
        """
        self.i_count += 1
        prob = norm.cdf(np.dot(alpha, self.thetas) + intercept)
        response = (np.random.uniform(0, 1) < prob).astype(int)
        self.d1_array[self.i_count] = (2*response-1)* alpha
        self.d2_array[self.i_count] = (2*response-1)* intercept
        self.s_array[self.i_count] = (np.sum(alpha**2) + 1)**0.5
        self.item_ids[self.i_count-1] = new_items
        self.item_responses[self.i_count-1] = response

    def eval_latent_traits(self, threshold: float):
        """Inference on Latent Traits"""
        i_ub = self.i_count + 1
        samples = self.sample_from_sun(self.d1_array[1:i_ub],
                                       self.s_array[1:i_ub],
                                       self.d2_array[1:i_ub],
                                       self.eval_samples)  # s by k
        for q in [0.25, 0.5, 0.75]:
            self.dist_stats[q][self.i_count] = np.quantile(samples, q, axis=0)
        self.dist_stats["mean"][self.i_count] = np.mean(samples, axis=0)
        cov_mat = np.cov(samples.T)
        self.dist_stats["cov"][self.i_count] = cov_mat[np.triu_indices_from(cov_mat)]
        linear_term = samples @ self.alphas.T + self.intercepts
        pred = norm.cdf(linear_term)  # s by j
        # in case we want to compute mi
        prev_selected_item = int(self.item_ids[self.i_count-1])
        prev_response = int(self.item_responses[self.i_count-1])
        if self.pred_type == "basic":
            prev_pred_mean = self.preds[self.i_count][prev_selected_item]
            self.preds[self.i_count][:self.m] = np.var(pred, axis=0)
            self.preds[self.i_count][-self.m:] = np.mean(pred, axis=0)
        else:
            prev_pred_mean = self.preds[prev_selected_item,0]
            self.preds[:, 0] = np.mean(pred, axis=0)
            self.preds[:, 1] = np.var(pred, axis=0)
            for i, q in enumerate(np.arange(0.1, 1, 0.1)):
                self.preds[:, (i + 2)] = np.quantile(pred, q=q, axis=0)
        if self.reward_type == "0-1":
            if np.max(np.diag(cov_mat)) > threshold:
                reward = -1
                done = False
            else:
                reward = 0
                done = True
        elif self.reward_type == "first0-1":
            if np.diag(cov_mat)[0] > threshold:
                reward = -1
                done = False
            else:
                reward = 0
                done =True
        elif self.reward_type == "first2-0-1":
            if np.max(np.diag(cov_mat)[:2]) > threshold:
                reward = -1
                done = False
            else:
                reward = 0
                done = True
            print(np.diag(cov_mat))
        elif self.reward_type == "first3-0-1":
            if np.max(np.diag(cov_mat)[:3]) > threshold:
                reward = -1
                done = False
            else:
                reward = 0
                done = True
        elif self.reward_type == "var-reduction":
            diagonal_indices = np.cumsum(np.array([0] + [self.k - i for i in range(self.k - 1)]))
            prev_vars = self.dist_stats["cov"][self.i_count - 1][diagonal_indices]
            reward = max(np.sum(prev_vars - np.diag(cov_mat)), 0)
            done = False
        elif self.reward_type == "log-cov":  # log covaru  determinant
            det_cur = np.linalg.det(cov_mat)
            cov_prev = np.zeros(cov_mat.shape)
            triu_indices = np.triu_indices_from(cov_prev)
            cov_prev[triu_indices] = self.dist_stats["cov"][self.i_count - 1]
            cov_prev = cov_prev + cov_prev.T - np.diag(np.diag(cov_prev))
            det_prev = np.linalg.det(cov_prev)
            reward = max(np.log(det_prev / det_cur), 0)
            done = False
        else: # mi
            prev_alpha = self.alphas[prev_selected_item]
            prev_intercept = self.intercepts[prev_selected_item]
            if prev_response == 1:
                p_sampled = np.mean(np.log(norm.cdf(samples@prev_alpha+prev_intercept)))
                mi = p_sampled - np.log(prev_pred_mean)
            else:
                p_sampled = np.mean(np.log(1-norm.cdf(samples @ prev_alpha + prev_intercept)))
                mi = p_sampled - np.log(1-prev_pred_mean)
            reward = max(0, mi)
            done = False
        return reward, done

    def sample_from_sun(self, d1i, s_array, d2_array, num_samples):
        """Sample from the Unified Skew Normal Distributions"""
        item_count = d1i.shape[0]
        cov_v0 = np.identity(self.k) - d1i.T @ (inv(d1i @ d1i.T + np.identity(item_count))) @ d1i
        cov_v1 = np.diag(1 / s_array) @ (d1i @ d1i.T + np.identity(item_count)) @ np.diag(1 / s_array)
        truc_lev = - np.diag(1 / s_array) @ (d2_array.reshape(-1, 1)).flatten()
        v0_s = self.rng.multivariate_normal(np.zeros(self.k), cov_v0, num_samples)
        # note if we set seed=self.random_state, then it will return the same thing for a given params!
        if cov_v1.shape[0] == 1: # we use scipy sampling if the dimension of truncated normal is 1
            v1_s = stats.truncnorm.rvs(truc_lev, np.inf, loc=0, scale=cov_v1, size=(1, num_samples))
        else:
            v1_s = TruncatedMVN(np.zeros(item_count), cov_v1, truc_lev,
                                np.ones_like(truc_lev) * np.inf, seed=None).sample(num_samples)
        linear_t = d1i.T @ (inv(d1i @ d1i.T + np.identity(item_count))) @ np.diag(s_array)
        return v0_s + (linear_t @ v1_s).T
