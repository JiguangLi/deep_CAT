import typing
import numpy as np
import statsmodels.api as sm
import multiprocessing
from .minimax_tilting import TruncatedMVN
from numpy.linalg import inv
import scipy.stats as stats


class OracleCAT:

    def __init__(
            self,
            item_response: np.ndarray,
            model_params: typing.Dict[str, np.ndarray],
            num_workers: int = 8,
            random_state: int = 42,
    ):
        """
        Create a Oracle CAT object that Estimate Test Takers Ability by Administering All Items in the Test Bank

            Args:
                item_response: Shape: (n, m), simulated item response data of each item in the test bank for each test taker
                model_params: dictionary with loading matrix "alphas", "intercepts", and latent factors "thetas"
                num_workers: number of cores for multiprocessing
                random_state: seed
            Raises:
                ValueError: if given inputs do not meet object expectations.
        """
        self.seed = random_state
        self.rng = np.random.default_rng(random_state)
        self.response = item_response
        self.thetas = model_params["thetas"]
        self.alphas = model_params["alphas"]
        self.intercepts = model_params["intercepts"]
        self.n, self.m = item_response.shape
        self.k = self.thetas.shape[1]
        self.num_workers = num_workers
        self.m = self.alphas.shape[0]

    def frequentist_inference(self):
        """ Return theta estimates by running probit regression with confidence intervals"""
        ci_lb, ci_ub, theta_est = np.zeros((self.n, self.k)), np.zeros((self.n, self.k)), np.zeros((self.n, self.k))
        for i in range(self.n):
            probit_model = sm.Probit(self.response[i], self.alphas, offset=self.intercepts)
            result = probit_model.fit(disp=0)
            theta_est[i] = result.params
            conf_intervals = result.conf_int()
            ci_lb[i] = conf_intervals[:, 0]
            ci_ub[i] = conf_intervals[:, 1]
        return {"theta_est": theta_est, "ci_lb": ci_lb, "ci_ub": ci_ub}

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

    def sun_inference(self, num_samples: int, ci=0.95, return_samples=False):
        """Bayesian Inference with Unified Skew-Normal Distribution"""
        d1_tensor = np.zeros((self.n, self.m, self.k))
        s_array = np.zeros((self.n, self.m))
        for i in range(self.n):
            d1_tensor[i] = (2*self.response[i]-1).reshape(-1,1) * self.alphas
            s_array[i] = (np.sum(d1_tensor[i] * d1_tensor[i], axis=1) + 1) ** 0.5
        d2_array = (2*self.response-1) * self.intercepts.reshape(1, -1)
        arguments = [(d1_tensor[i], s_array[i], d2_array[i], num_samples) for i in range(self.n)]
        with multiprocessing.Pool(self.num_workers) as pool:
            out = pool.starmap(self.sample_from_sun, arguments)
        samples = np.zeros((self.n, num_samples, self.k))
        for i in range(self.n):
            samples[i] = out[i]
        pos_mean = np.mean(samples, axis=1)
        ci_lb = np.quantile(samples, (1 - ci) / 2, axis=1)
        ci_ub = np.quantile(samples, ci + (1 - ci) / 2, axis=1)
        if return_samples:
            result = {"pos_mean": pos_mean, "ci_lb": ci_lb, "ci_ub": ci_ub, "samples": samples}
        else:
            result = {"ci_lb": ci_lb, "ci_ub": ci_ub, "pos_mean": pos_mean}
        return result







