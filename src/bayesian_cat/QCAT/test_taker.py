import numpy as np
from numpy.linalg import inv
import scipy.stats as stats
from scipy.stats import norm
import multiprocessing
from .minimax_tilting import TruncatedMVN


class TestTakers:

    def __init__(
            self,
            thetas: np.ndarray,
            max_item: int,
            num_workers: int,
            random_state: int
    ):
        """
        Create Test Taker Object with true ability theta

            Args:
                thetas: n by k array true latent factor
                max_item: maximum amount of items administered to this test taker
                num_workers: number of cores for parallel computing
                random_state: seed
        """
        self.rng = np.random.default_rng(random_state)
        self.random_state = random_state
        self.num_workers = num_workers
        self.thetas = thetas
        max_item = max_item + 1  # since we filled index zero with zeros
        self.max_item = max_item
        self.k = thetas.shape[1]
        self.n = thetas.shape[0]
        self.pos_mean = np.zeros((max_item, self.n, self.k))
        self.d1_tensor = np.zeros((self.n, max_item, self.k))
        self.d2_array = np.zeros((self.n, max_item))
        self.s_array = np.zeros((self.n, max_item))
        self.i_count = 0
        self.item_ids = np.ones((self.n, max_item))*(-1)
        self.item_ids = self.item_ids.astype(int)
        self.item_responses = np.zeros((self.n, max_item))
        cov_params = int((1 + self.k) * self.k / 2)
        self.dist_stats = {0.25: np.zeros((self.n, max_item, self.k)),
                           0.5: np.zeros((self.n, max_item,self.k)),
                           0.75: np.zeros((self.n, max_item,self.k)),
                           "mean": np.zeros((self.n, max_item,self.k)),
                           "cov": np.zeros((self.n, max_item,cov_params))}
        self.dist_stats[0.25][:, 0, :] = -0.67448
        self.dist_stats[0.5][:, 0, :] = 0
        self.dist_stats[0.75][:, 0, :] = 0.67448
        cov_mat = np.identity(self.k)
        self.dist_stats["cov"][:, 0, :] = cov_mat[np.triu_indices_from(cov_mat)]

    def answer_items(self, alphas: np.ndarray, intercepts: np.ndarray, new_items: np.ndarray):
        """Simulate the process of answering an item given its loading and intercept
            Args:
                alphas: n by k dimensional array
                intercepts: n-dimensional array
                new_items: n-dimensional array
        """
        self.i_count += 1
        linear_term = np.sum(alphas*self.thetas, axis=1) + intercepts
        prob_vec = norm.cdf(linear_term)
        response_vec = (np.random.uniform(0, 1, self.n) < prob_vec).astype(int)
        self.d1_tensor[:, self.i_count, :] = (2*response_vec-1).reshape(-1,1) * alphas
        self.d2_array[:, self.i_count] = (2*response_vec-1) * intercepts
        d1_sub_mat = self.d1_tensor[:, self.i_count, :]
        self.s_array[:, self.i_count] = (np.sum(d1_sub_mat * d1_sub_mat, axis=1) + 1)**0.5
        self.item_ids[:, self.i_count-1] = new_items
        self.item_responses[:, self.i_count-1] = response_vec

    def answer_true_responses(self, alphas: np.ndarray, intercepts: np.ndarray, new_items: np.ndarray,
                              response_vec: np.ndarray):
        """Simulate the process of answering an item given its loading and intercept
            Args:
                alphas: n by k dimensional array
                intercepts: n-dimensional array
                new_items: n-dimensional array
        """
        self.i_count += 1
        self.d1_tensor[:, self.i_count, :] = (2 * response_vec - 1).reshape(-1, 1) * alphas
        self.d2_array[:, self.i_count] = (2 * response_vec - 1) * intercepts
        d1_sub_mat = self.d1_tensor[:, self.i_count, :]
        self.s_array[:, self.i_count] = (np.sum(d1_sub_mat * d1_sub_mat, axis=1) + 1) ** 0.5
        self.item_ids[:, self.i_count - 1] = new_items
        self.item_responses[:, self.i_count - 1] = response_vec

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

    def get_factor_samples(self, num_samples):
        """Draw from Latent Factor Posterior Distribution
        :returns
            samples: with shape (n, num_samples, k)
        """
        i_ub = self.i_count+1
        arguments = [(self.d1_tensor[i, 1:i_ub, :], self.s_array[i, 1:i_ub], self.d2_array[i, 1:i_ub], num_samples) for i in range(self.n)]
        with multiprocessing.Pool(self.num_workers) as pool:
            out = pool.starmap(self.sample_from_sun, arguments)
        samples = np.zeros((self.n, num_samples, self.k))
        for i in range(self.n):
            samples[i] = out[i]
        return samples
