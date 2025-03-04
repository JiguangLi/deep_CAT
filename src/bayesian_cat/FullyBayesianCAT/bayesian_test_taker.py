import numpy as np
from numpy.linalg import inv
import scipy.stats as stats
from scipy.stats import norm
import multiprocessing
from .minimax_tilting import TruncatedMVN


class BayesianTestTakers:

    def __init__(
            self,
            thetas: np.ndarray,
            pos_num_samples: int,
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
        self.pos_num_samples = pos_num_samples
        self.max_item = max_item
        self.k = thetas.shape[1]
        self.n = thetas.shape[0]
        self.d1_tensor = np.zeros((self.n, self.pos_num_samples, max_item, self.k))
        self.d2_array = np.zeros((self.n, self.pos_num_samples, max_item))
        self.s_array = np.zeros((self.n, self.pos_num_samples, max_item))
        self.i_count = -1
        self.item_ids = np.ones((self.n, max_item))*(-1)
        self.item_responses = np.zeros((self.n, max_item))

    # def answer_items(self, alphas: np.ndarray, intercepts: np.ndarray, new_items: np.ndarray):
    #     """Simulate the process of answering an item given its loading and intercept
    #         Args:
    #             alphas: s * n by k dimensional array
    #             intercepts: s* n-dimensional array
    #             new_items: n-dimensional array
    #     """
    #     self.i_count += 1
    #     linear_term = np.sum(alphas* self.thetas, axis=2) + intercepts
    #     prob_vec = np.mean(norm.cdf(linear_term), axis=0)
    #     response_vec = (np.random.uniform(0, 1, self.n) < prob_vec).astype(int)
    #     self.d1_tensor[:, :, self.i_count, :] = np.transpose(alphas * (2 * response_vec - 1).reshape(1, -1),
    #                                                          axes=(1, 0, 2))  # n * s * k
    #     self.d2_array[:, :, self.i_count] = intercepts.T * ((2 * response_vec - 1)[:, np.newaxis])  # n by s
    #     d1_sub_mat = self.d1_tensor[:, :, self.i_count, :]  # n by s by k
    #     self.s_array[:, :, self.i_count] = (np.sum(d1_sub_mat * d1_sub_mat, axis=2) + 1) ** 0.5  # n by s
    #     self.item_ids[:, self.i_count] = new_items
    #     self.item_responses[:, self.i_count] = response_vec

    def answer_true_responses(self, alphas: np.ndarray, intercepts: np.ndarray, new_items: np.ndarray,
                                response_vec: np.ndarray):
        """Simulate the process of answering an item giv
        en its loading and intercept
            Args:
                alphas: s* n by k dimensional array
                intercepts: s * n-dimensional array
                new_items: n-dimensional array
        """
        self.i_count += 1
        self.d1_tensor[:, :, self.i_count, :] = np.transpose(alphas * (2*response_vec-1)[np.newaxis, :, np.newaxis], axes = (1, 0, 2)) # n * s * k
        self.d2_array[:, :, self.i_count] = intercepts.T * ((2*response_vec-1)[:, np.newaxis]) # n by s
        d1_sub_mat = self.d1_tensor[:, :, self.i_count, :] # n by s by k
        self.s_array[:, :, self.i_count] = (np.sum(d1_sub_mat * d1_sub_mat, axis=2) + 1) ** 0.5 # n by s
        self.item_ids[:, self.i_count] = new_items
        self.item_responses[:, self.i_count] = response_vec

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
            samples: with shape (n, pos_num_samples, num_samples, k)
        """
        i_ub = self.i_count+1
        arguments = [(self.d1_tensor[i, j, :i_ub, :], self.s_array[i, j,:i_ub],
                      self.d2_array[i, j, :i_ub], num_samples)
                     for i in range (self.n)
                     for j in range(self.pos_num_samples)]
        with multiprocessing.Pool(self.num_workers) as pool:
            out = pool.starmap(self.sample_from_sun, arguments)
        out = np.array(out)
        out = out.reshape(self.n, self.pos_num_samples, num_samples, self.k)
        return out

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

    def get_exact_oracle_samples(self, p_idx: int, num_samples: int, alphas: np.ndarray, intercepts: np.ndarray,
                                 u_hat: np.ndarray):
        """Draw From Future Factor Posterior Distribution Conditional on New responses (for one person only)
         :returns
            samples: with shape (num_samples, k)
        """
        ub = self.i_count + 1
        d1_aug = np.vstack((self.d1_tensor[p_idx, :ub, :], np.diag(2*u_hat-1)@alphas))
        d2_aug = np.concatenate((self.d2_array[p_idx, :ub], intercepts * (2 * u_hat - 1)))
        s_aug = np.concatenate((self.s_array[p_idx, :ub], (np.sum(alphas ** 2, axis=1) + 1) ** 0.5))
        samples = self.sample_from_sun(d1_aug, s_aug, d2_aug,  num_samples)
        return samples

    def get_sir_reweighted_samples(self, posterior_samples: np.ndarray, sir_samples: int, alphas: np.ndarray,
                                   intercepts: np.ndarray):
        """Sampling-Resampling Posterior Reweighting


        @ Args:
            posterior_samples: num_pos_samples by mc_samples by k
        @ return num_items* pos_samples  * sir_samples * k
        """
        num_pos_samples, mc_samples, k = posterior_samples.shape
        num_items = alphas.shape[1]
        # compute likelihood
        prob1 = np.einsum('pak,pmk->pam', alphas, posterior_samples)  # pos_samples by a by m
        prob1 += intercepts[:, :, np.newaxis] # pos_samples by a by m
        prob1 = norm.cdf(prob1) # pos_samples by a by m
        pred_hat = np.mean(prob1, axis=(0, 2)) # a-dim vector
        prob0 = 1 - prob1
        # reweighting
        theta1_sampled = np.zeros((num_items, num_pos_samples, sir_samples, k))
        theta0_sampled = np.zeros((num_items, num_pos_samples, sir_samples, k))
        for j in range(num_items):
            # if next is 1
            temp_prob1 = prob1[:, j, :] # pos_samples by mc_samples
            temp_weight1 = temp_prob1/np.sum(temp_prob1)
            flat_indices1 = np.random.choice(num_pos_samples * sir_samples, size=num_pos_samples * sir_samples,
                                            replace=True, p=temp_weight1.flatten())
            row_indices, col_indices = np.unravel_index(flat_indices1, (num_pos_samples, sir_samples))
            samples1 = posterior_samples[row_indices, col_indices] # (pos_samples * num_samples) * k
            theta1_sampled[j] = samples1.reshape(num_pos_samples, sir_samples, k)
            # if next is zero
            temp_prob0 = prob0[:, j, :]  # pos_samples by mc_samples
            temp_weight0 = temp_prob0 / np.sum(temp_prob0)
            flat_indices0 = np.random.choice(num_pos_samples * sir_samples, size=num_pos_samples * sir_samples,
                                             replace=True, p=temp_weight0.flatten())
            row_indices, col_indices = np.unravel_index(flat_indices0, (num_pos_samples, sir_samples))
            samples0 = posterior_samples[row_indices, col_indices]  # (pos_samples * num_samples) * k
            theta0_sampled[j] = samples0.reshape(num_pos_samples, sir_samples, k)
        return {"0": theta0_sampled , "1": theta1_sampled, "pred_hat": pred_hat}


    def get_pos_mean(self, num_samples: int):
        """Get posterior samples and Means

        return samples and mean
        """
        samples = self.get_factor_samples(num_samples)
        cur_mean = np.mean(samples, axis=(1, 2))
        return samples, cur_mean

    def get_pos_pred(self, item_idx: int, num_samples: int, alphas: np.ndarray, intercepts: np.ndarray):
        """Get posterior prediction probabilities"""
        samples = self.get_factor_samples(num_samples) # (n, s* num_samples, k)
        m = alphas.shape[0]
        preds = np.zeros((self.n, m-item_idx))
        for i in range(self.n):
            samples_i = samples[i].T # k by num_samples
            all_items = np.arange(m)
            avail_items = all_items[~np.in1d(all_items, self.item_ids[i, :item_idx])]
            preds[i] = np.mean(norm.cdf(alphas[avail_items]@samples_i + intercepts[avail_items].reshape(-1, 1)), axis=1)
        return preds




