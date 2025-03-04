import numpy as np
from numpy.linalg import inv
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
        self.max_item = max_item
        self.k = thetas.shape[1]
        self.n = thetas.shape[0]
        self.pos_mean = np.zeros((max_item, self.n, self.k))
        self.d1_tensor = np.zeros((self.n, max_item, self.k))
        self.d2_array = np.zeros((self.n, max_item))
        self.s_array = np.zeros((self.n, max_item))
        self.i_count = -1
        self.item_ids = np.ones((self.n, max_item))*(-1)

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
        self.item_ids[:, self.i_count] = new_items

    def sample_from_sun(self, d1i, s_array, d2_array, num_samples):
        """Sample from the Unified Skew Normal Distributions"""
        item_count = self.i_count + 1
        cov_v0 = np.identity(self.k) - d1i.T @ (inv(d1i @ d1i.T + np.identity(item_count))) @ d1i
        cov_v1 = np.diag(1 / s_array) @ (d1i @ d1i.T + np.identity(item_count)) @ np.diag(1 / s_array)
        truc_lev = - np.diag(1 / s_array) @ (d2_array.reshape(-1, 1)).flatten()
        v0_s = self.rng.multivariate_normal(np.zeros(self.k), cov_v0, num_samples)
        # note if we set seed=self.random_state, then it will return the same thing for a given params!
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
        arguments = [(self.d1_tensor[i, :i_ub, :], self.s_array[i, :i_ub], self.d2_array[i, :i_ub], num_samples) for i in range(self.n)]
        with multiprocessing.Pool(self.num_workers) as pool:
            out = pool.starmap(self.sample_from_sun, arguments)
        samples = np.zeros((self.n, num_samples, self.k))
        for i in range(self.n):
            samples[i] = out[i]
        return samples

    def get_next_factor_samples(self, p_idx: int, cond_rep: int, num_samples: int,
                                alphas: np.ndarray, intercepts: np.ndarray):
        """Draw From Future Factor Posterior Distribution Conditional on New responses (for one person only)
         :returns
            samples: with shape (avail_items, num_samples, k)
        """
        ub = self.i_count + 1
        d1_aug = np.vstack((self.d1_tensor[p_idx, :ub, :], alphas*(2*cond_rep-1)))
        d2_aug = np.concatenate((self.d2_array[p_idx, :ub], intercepts*(2*cond_rep-1)))
        s_aug = np.concatenate((self.s_array[p_idx, :ub], (np.sum(alphas**2, axis=1)+1)**0.5))
        arguments = [(d1_aug[np.append(np.arange(ub), j), :], s_aug[np.append(np.arange(ub), j)],
                      d2_aug[np.append(np.arange(ub), j)], num_samples) for j in range(ub, ub+alphas.shape[0])]
        with multiprocessing.Pool(self.num_workers) as pool:
            out = pool.starmap(self.sample_from_sun, arguments)
        samples = np.zeros((alphas.shape[0], num_samples, self.k))
        for i in range(alphas.shape[0]):
            samples[i] = out[i]
        return samples

    def get_pos_mean(self, item_idx: int, num_samples: int):
        """Get posterior Means"""
        samples = self.get_factor_samples(num_samples)
        cur_mean = np.mean(samples, axis=1)
        self.pos_mean[item_idx] = cur_mean

    def get_pos_pred(self, item_idx: int, num_samples: int, alphas: np.ndarray, intercepts: np.ndarray):
        """Get posterior prediction probabilities"""
        samples = self.get_factor_samples(num_samples) # (n, num_samples, k)
        preds = np.zeros((self.n, self.max_item-item_idx))
        for i in range(self.n):
            samples_i = samples[i].T # k by num_samples
            all_items = np.arange(self.max_item)
            avail_items = all_items[~np.in1d(all_items, self.item_ids[i, :item_idx])]
            preds[i] = np.mean(norm.cdf(alphas[avail_items]@samples_i + intercepts[avail_items].reshape(-1, 1)), axis=1)
        return preds




