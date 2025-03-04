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
        self.max_item = max_item
        self.k = thetas.shape[1]
        self.n = thetas.shape[0]
        self.pos_mean = np.zeros((max_item, self.n, self.k))
        self.d1_tensor = np.zeros((self.n, max_item, self.k))
        self.d2_array = np.zeros((self.n, max_item))
        self.s_array = np.zeros((self.n, max_item))
        self.i_count = -1
        self.item_ids = np.ones((self.n, max_item))*(-1)
        self.item_responses = np.zeros((self.n, max_item))

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
        self.item_responses[:, self.i_count] = response_vec

    def answer_true_responses(self, alphas: np.ndarray, intercepts: np.ndarray, new_items: np.ndarray,
                                response_vec: np.ndarray):
        """Simulate the process of answering an item given its loading and intercept
            Args:
                alphas: n by k dimensional array
                intercepts: n-dimensional array
                new_items: n-dimensional array
        """
        self.i_count += 1
        self.d1_tensor[:, self.i_count, :] = (2*response_vec-1).reshape(-1,1) * alphas
        self.d2_array[:, self.i_count] = (2*response_vec-1) * intercepts
        d1_sub_mat = self.d1_tensor[:, self.i_count, :]
        self.s_array[:, self.i_count] = (np.sum(d1_sub_mat * d1_sub_mat, axis=1) + 1)**0.5
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
        """Sampling-Resampling Posterior Reweighting"""
        sir_large_samples = posterior_samples.shape[0]
        prob1 = norm.cdf(alphas @ posterior_samples.T + intercepts.reshape(-1,1)) # j * sir_large_samples
        threshold = 1e-10
        prob1 = np.where(prob1 > (1 - threshold), 1 - threshold, prob1)
        prob1 = np.where(prob1 < threshold, threshold, prob1)
        prob0 = 1 - prob1
        weight1 = prob1/(np.sum(prob1, axis=1).reshape(-1,1)) # j * sir_large_samples
        weight0 = prob0/(np.sum(prob0, axis=1).reshape(-1, 1))
        num_items, k = alphas.shape
        theta1_sampled, theta0_sampled = np.zeros((num_items, sir_samples, k)), np.zeros((num_items, sir_samples, k))
        for j in range(num_items):
            indices0 = np.random.choice(a=sir_large_samples, size=sir_samples, p=weight0[j])
            indices1 = np.random.choice(a=sir_large_samples, size=sir_samples, p=weight1[j])
            theta0_sampled[j] = posterior_samples[indices0]
            theta1_sampled[j] = posterior_samples[indices1]
        return {"0": theta0_sampled , "1": theta1_sampled}

    def get_oracle_samples(self, posterior_samples: np.ndarray, sir_samples: int, alphas: np.ndarray,
                                     intercepts: np.ndarray, u_hat: np.ndarray):
        """Sampling-Resampling Posterior to get the oracle posterior
        u_hat: 1d array having num_of avaliable items
        """
        threshold = 1e-10
        sir_large_samples = posterior_samples.shape[0]
        prob1 = norm.cdf(alphas @ posterior_samples.T + intercepts.reshape(-1,1)) # j * sir_large_samples
        prob1 = np.where(prob1 > (1-threshold), 1-threshold, prob1)
        prob1 = np.where(prob1 < threshold, threshold, prob1)
        prob0 = 1 - prob1 # j * sir_large_sample
        num_items, k = alphas.shape
        oracle_universal_weight = np.empty_like(prob1)
        oracle_universal_weight[u_hat == 0] = prob0[u_hat == 0]
        oracle_universal_weight[u_hat == 1] = prob1[u_hat == 1]
        oracle_universal_weight = np.sum(np.log(oracle_universal_weight), axis=0)
        median_weight = np.median(oracle_universal_weight)
        oracle_universal_weight = np.exp(oracle_universal_weight - median_weight)
        oracle_universal_weight = oracle_universal_weight/np.sum(oracle_universal_weight)
        temp_indices = np.random.choice(a=sir_large_samples, size=sir_samples, p=oracle_universal_weight)
        universal_oracle_samples = posterior_samples[temp_indices]
        theta1_sampled, theta0_sampled = np.zeros((num_items, sir_samples, k)), np.zeros((num_items, sir_samples, k))
        for j in range(num_items):
            if u_hat[j] == 0:
                theta0_sampled[j] = universal_oracle_samples
                reweight = prob1[j] / np.where(prob0[j] < threshold, threshold, prob0[j])
                new_weight = oracle_universal_weight * reweight
                new_weight = new_weight / np.sum(new_weight)
                #print(oracle_universal_weight[nan_indices[0]]  ,(prob1[j] / prob0[j])[nan_indices[0]],new_weight[nan_indices[0]])
                indices1 = np.random.choice(a=sir_large_samples, size=sir_samples, p=new_weight)
                theta1_sampled[j] = posterior_samples[indices1]
            else:
                theta1_sampled[j] = universal_oracle_samples
                reweight = prob0[j] / np.where(prob1[j] < threshold, threshold, prob1[j])
                new_weight = oracle_universal_weight * reweight
                new_weight = new_weight / np.sum(new_weight)
                indices0 = np.random.choice(a=sir_large_samples, size=sir_samples, p=new_weight)
                theta0_sampled[j] = posterior_samples[indices0]
        return {"0": theta0_sampled , "1": theta1_sampled}

    def get_pos_mean(self, item_idx: int, num_samples: int):
        """Get posterior Means"""
        samples = self.get_factor_samples(num_samples)
        cur_mean = np.mean(samples, axis=1)
        self.pos_mean[item_idx] = cur_mean

    def get_pos_pred(self, item_idx: int, num_samples: int, alphas: np.ndarray, intercepts: np.ndarray):
        """Get posterior prediction probabilities"""
        samples = self.get_factor_samples(num_samples) # (n, num_samples, k)
        m = alphas.shape[0]
        preds = np.zeros((self.n, m-item_idx))
        for i in range(self.n):
            samples_i = samples[i].T # k by num_samples
            all_items = np.arange(m)
            avail_items = all_items[~np.in1d(all_items, self.item_ids[i, :item_idx])]
            preds[i] = np.mean(norm.cdf(alphas[avail_items]@samples_i + intercepts[avail_items].reshape(-1, 1)), axis=1)
        return preds




