import numpy as np
from numpy.linalg import inv
import scipy.stats as stats
import multiprocessing
from .minimax_tilting import TruncatedMVN


def sample_from_sun(d1i, s_array, d2_array, num_samples):
    """Sample from the Unified Skew Normal Distributions"""
    item_count, k = d1i.shape
    cov_v0 = np.identity(k) - d1i.T @ (inv(d1i @ d1i.T + np.identity(item_count))) @ d1i
    cov_v1 = np.diag(1 / s_array) @ (d1i @ d1i.T + np.identity(item_count)) @ np.diag(1 / s_array)
    truc_lev = - np.diag(1 / s_array) @ (d2_array.reshape(-1, 1)).flatten()
    v0_s = np.random.multivariate_normal(np.zeros(k), cov_v0, num_samples)
    # note if we set seed=self.random_state, then it will return the same thing for a given params!
    if cov_v1.shape[0] == 1:  # we use scipy sampling if the dimension of truncated normal is 1
        v1_s = stats.truncnorm.rvs(truc_lev, np.inf, loc=0, scale=cov_v1, size=(1, num_samples))
    else:
        v1_s = TruncatedMVN(np.zeros(item_count), cov_v1, truc_lev,
                            np.ones_like(truc_lev) * np.inf, seed=None).sample(num_samples)
    linear_t = d1i.T @ (inv(d1i @ d1i.T + np.identity(item_count))) @ np.diag(s_array)
    return v0_s + (linear_t @ v1_s).T


def obtain_sun_samples(num_samples: int, response: np.ndarray, alphas: np.ndarray, intercepts: np.ndarray,
                       item_selection: np.ndarray, num_workers =8):
    """Sample from SUN Given Item Selections

    Args:
        num_samples: num_samples
        response: n by temp_m (number of selected items)
        alphas: m by k
        intercepts: m-dim vec
        item_selection: n by temp_m
    """
    n, temp_m = response.shape
    m, k = alphas.shape
    d1_tensor = np.zeros((n, temp_m, k))
    d2_array = np.zeros((n, temp_m))
    s_array = np.zeros((n, temp_m))
    for j in range(temp_m):
        d1_tensor[:, j, :] = (2 * response[:, j] - 1).reshape(-1, 1) * alphas[item_selection[:, j]]
        d2_array[:, j] = (2 * response[:, j] - 1) * intercepts[item_selection[:, j]]
        d1_sub_mat = d1_tensor[:, j, :]
        s_array[:, j] = (np.sum(d1_sub_mat * d1_sub_mat, axis=1) + 1) ** 0.5
    arguments = [(d1_tensor[i], s_array[i], d2_array[i], num_samples) for i in range(n)]
    with multiprocessing.Pool(num_workers) as pool:
        out = pool.starmap(sample_from_sun, arguments)
    samples = np.zeros((n, num_samples, k))
    for i in range(n):
        samples[i] = out[i]
    return samples


def sample_from_sun_all(num_samples, d1_tensor, d2_array,  s_array, num_workers =8):
    """Sample from SUN Given all Test TakersParameters """
    n, m, k = d1_tensor.shape
    arguments = [(d1_tensor[i], s_array[i], d2_array[i], num_samples) for i in range(n)]
    with multiprocessing.Pool(num_workers) as pool:
        out = pool.starmap(sample_from_sun, arguments)
    samples = np.zeros((n, num_samples, k))
    for i in range(n):
        samples[i] = out[i]
    return samples

