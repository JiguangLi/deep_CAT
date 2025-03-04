import pathlib
import numpy as np
import yaml
import argparse
import typing
import pickle
from numpy.linalg import inv
import scipy.stats as stats
from scipy.stats import norm
from bayesian_cat import TruncatedMVN, OnlineQNetworkV4
from tqdm import tqdm
import torch


def parse_args(verbose: bool = False) -> typing.Dict[str, typing.Any]:
    """Get command line arguments, with defaults set for local testing."""
    # parse command line arguments first
    parser = argparse.ArgumentParser("Testing MCAT Speeds w/o Parallel Computing")
    parser.add_argument("--config_filepath",
                        default= pathlib.Path.home().joinpath(pathlib.Path("bayesian-cat", "config", "config.yaml")))
    parser.add_argument("--verbose", default=True)
    parser.add_argument("--input_data_name", default="x09_s01_mixed_grid_500_200_5_4.pickle")
    parser.add_argument("--max_items", default=50, help="after how many items to stop")
    parser.add_argument("--sir_start", default=1, help="after how many items to do posterior reweighting")
    parser.add_argument("--mc_samples", default=500)
    parser.add_argument("--sir_large_samples", default=750)
    parser.add_argument("--sir_samples", default=500)
    parser.add_argument("--dim", default=3)
    parser.add_argument("--num_workers", default=8)
    parser.add_argument("--seed", default=42)
    arguments = vars(parser.parse_args())
    # get remaining arguments from config
    with open(arguments["config_filepath"]) as config_file:
        config = yaml.full_load(config_file)
    arguments.update(config)
    # fin
    if verbose:
        print(arguments)
    return arguments


def sample_from_sun(d1i, s_array, d2_array, num_samples):
    """Sample from the Unified Skew Normal Distributions"""
    item_count = d1i.shape[0]
    k = len(d1i[0])
    cov_v0 = np.identity(k) - d1i.T @ (inv(d1i @ d1i.T + np.identity(item_count))) @ d1i
    cov_v1 = np.diag(1 / s_array) @ (d1i @ d1i.T + np.identity(item_count)) @ np.diag(1 / s_array)
    truc_lev = - np.diag(1 / s_array) @ (d2_array.reshape(-1, 1)).flatten()
    v0_s = np.random.multivariate_normal(np.zeros(k), cov_v0, num_samples)
    # note if we set seed=self.random_state, then it will return the same thing for a given params!
    if cov_v1.shape[0] == 1: # we use scipy sampling if the dimension of truncated normal is 1
        v1_s = stats.truncnorm.rvs(truc_lev, np.inf, loc=0, scale=cov_v1, size=(1, num_samples))
    else:
        v1_s = TruncatedMVN(np.zeros(item_count), cov_v1, truc_lev,
                            np.ones_like(truc_lev) * np.inf, seed=None).sample(num_samples)
    linear_t = d1i.T @ (inv(d1i @ d1i.T + np.identity(item_count))) @ np.diag(s_array)
    return v0_s + (linear_t @ v1_s).T

def get_sir_reweighted_samples(posterior_samples: np.ndarray, sir_samples: int, alphas: np.ndarray,
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

def pick_starting_item(alphas, intercepts):
    """Pick the starting item from the test bank
    """
    icpt_40 = np.quantile(intercepts, 0.4)
    icpt_60 = np.quantile(intercepts, 0.6)
    indices = np.where((intercepts >= icpt_40) & (intercepts <= icpt_60))[0]
    abs_sum = np.sum(np.abs(alphas[indices]), axis=1)
    upper_quantile_value = np.quantile(abs_sum, 0.75)
    arg_max_upper_quantile = np.argmin(np.abs(abs_sum - upper_quantile_value))
    return indices[arg_max_upper_quantile]

def test_kl_eap_speed(theta, test_params, mc_samples, max_items):
    alphas, intercepts = test_params["alphas"], test_params["intercepts"]
    m, k = alphas.shape
    temp_item = pick_starting_item(alphas, intercepts)
    d1_array = np.zeros((max_items, k))
    d2_array = np.zeros(max_items)
    s_array = np.zeros(max_items)
    item_selections = np.zeros(max_items)
    for j in tqdm(range(max_items)):
        # update posterior params
        item_selections[j] = temp_item
        temp_alpha = alphas[temp_item]
        temp_intercept = intercepts[temp_item]
        linear_term = np.sum(alphas * theta) + temp_intercept
        prob = norm.cdf(linear_term)
        response = (np.random.uniform(0, 1, 1) < prob).astype(int)
        d1_array[j] = (2*response-1)*temp_alpha
        d2_array[j] = (2*response-1)* temp_intercept
        s_array[j] = (np.sum(d1_array[j]*d1_array[j]) + 1)**0.5
        # item selection
        i_ub = j+1
        new_samples =  sample_from_sun(d1_array[:i_ub], s_array[:i_ub], d2_array[:i_ub], mc_samples)
        theta_hat = np.mean(new_samples, axis=0)
        avail_items = np.setdiff1d(np.arange(m), item_selections[:i_ub])
        p_hat = norm.cdf(alphas[avail_items, :] @ theta_hat + intercepts[avail_items])  # j-dim vec
        p_sampled = norm.cdf(alphas[avail_items, :] @ new_samples.T + intercepts[avail_items].reshape(-1,1))  # j * mc_samples
        kl_info = p_hat * np.mean(
            np.log(1 / p_sampled * p_hat.reshape(-1, 1)), axis=1) + (1 - p_hat) * np.mean(
            np.log(1 / (1 - p_sampled) * (1 - p_hat).reshape(-1, 1)), axis=1)
        temp_item = avail_items[np.argmax(kl_info)]
    return item_selections


def test_kl_pos_speed(theta, test_params, mc_samples, max_items):
    alphas, intercepts = test_params["alphas"], test_params["intercepts"]
    m, k = alphas.shape
    temp_item = pick_starting_item(alphas, intercepts)
    d1_array = np.zeros((max_items, k))
    d2_array = np.zeros(max_items)
    s_array = np.zeros(max_items)
    item_selections = np.zeros(max_items)
    for j in tqdm(range(max_items)):
        # update posterior params
        item_selections[j] = temp_item
        temp_alpha = alphas[temp_item]
        temp_intercept = intercepts[temp_item]
        linear_term = np.sum(alphas * theta) + temp_intercept
        prob = norm.cdf(linear_term)
        response = (np.random.uniform(0, 1, 1) < prob).astype(int)
        d1_array[j] = (2 * response - 1) * temp_alpha
        d2_array[j] = (2 * response - 1) * temp_intercept
        s_array[j] = (np.sum(d1_array[j] * d1_array[j]) + 1) ** 0.5
        # item selection
        i_ub = j + 1
        new_samples = sample_from_sun(d1_array[:i_ub], s_array[:i_ub], d2_array[:i_ub], mc_samples)
        avail_items = np.setdiff1d(np.arange(m), item_selections[:i_ub])
        p_sampled = norm.cdf(alphas[avail_items, :] @ new_samples.T + intercepts[avail_items].reshape(-1, 1))  # j * mc_samples
        pred_hat = np.mean(p_sampled, axis=1)
        kl_term1 = pred_hat * np.mean(np.log(1 / p_sampled * pred_hat.reshape(-1, 1)), axis=1)
        kl_term2 = (1 - pred_hat) * np.mean(np.log(1 / (1 - p_sampled) * (1 - pred_hat).reshape(-1, 1)), axis=1)
        temp_item = avail_items[np.argmax(kl_term1 + kl_term2)]
    return item_selections


def test_mi_sir_speed(theta, test_params, mc_samples, sir_large_samples, max_items):
    alphas, intercepts = test_params["alphas"], test_params["intercepts"]
    m, k = alphas.shape
    temp_item = pick_starting_item(alphas, intercepts)
    d1_array = np.zeros((max_items, k))
    d2_array = np.zeros(max_items)
    s_array = np.zeros(max_items)
    item_selections = np.zeros(max_items)
    for j in tqdm(range(max_items)):
        # update posterior params
        item_selections[j] = temp_item
        temp_alpha = alphas[temp_item]
        temp_intercept = intercepts[temp_item]
        linear_term = np.sum(alphas * theta) + temp_intercept
        prob = norm.cdf(linear_term)
        response = (np.random.uniform(0, 1, 1) < prob).astype(int)
        d1_array[j] = (2 * response - 1) * temp_alpha
        d2_array[j] = (2 * response - 1) * temp_intercept
        s_array[j] = (np.sum(d1_array[j] * d1_array[j]) + 1) ** 0.5
        # item selection
        i_ub = j + 1
        new_samples = sample_from_sun(d1_array[:i_ub], s_array[:i_ub], d2_array[:i_ub], mc_samples)
        avail_items = np.setdiff1d(np.arange(m), item_selections[:i_ub])
        p_sampled = norm.cdf(
            alphas[avail_items, :] @ new_samples.T + intercepts[avail_items].reshape(-1, 1))  # j * mc_samples
        pred_hat = np.mean(p_sampled, axis=1)
        reweighted_samples = get_sir_reweighted_samples(new_samples, sir_large_samples,
                                                                     alphas[avail_items, :],
                                                                     intercepts[avail_items])
        p_sampled0 = 1 - norm.cdf(
            np.squeeze(np.matmul(reweighted_samples["0"], alphas[avail_items, :, np.newaxis]), axis=-1
                       ) + intercepts[avail_items].reshape(-1, 1))  # j by mc samples
        p_sampled1 = norm.cdf(
            np.squeeze(np.matmul(reweighted_samples["1"], alphas[avail_items, :, np.newaxis]), axis=-1
                       ) + intercepts[avail_items].reshape(-1, 1))  # j by mc samples
        mi_term1 = pred_hat * np.mean(np.log(p_sampled1 / pred_hat.reshape(-1, 1)), axis=1)
        mi_term2 = (1 - pred_hat) * np.mean(np.log(p_sampled0 / (1 - pred_hat).reshape(-1, 1)), axis=1)
        temp_item= avail_items[np.argmax(mi_term1 + mi_term2)]
    return item_selections


def test_prediction_ve_speed(theta, test_params, mc_samples, max_items):
    alphas, intercepts = test_params["alphas"], test_params["intercepts"]
    m, k = alphas.shape
    temp_item = pick_starting_item(alphas, intercepts)
    d1_array = np.zeros((max_items, k))
    d2_array = np.zeros(max_items)
    s_array = np.zeros(max_items)
    item_selections = np.zeros(max_items)
    for j in tqdm(range(max_items)):
        # update posterior params
        item_selections[j] = temp_item
        temp_alpha = alphas[temp_item]
        temp_intercept = intercepts[temp_item]
        linear_term = np.sum(alphas * theta) + temp_intercept
        prob = norm.cdf(linear_term)
        response = (np.random.uniform(0, 1, 1) < prob).astype(int)
        d1_array[j] = (2 * response - 1) * temp_alpha
        d2_array[j] = (2 * response - 1) * temp_intercept
        s_array[j] = (np.sum(d1_array[j] * d1_array[j]) + 1) ** 0.5
        # item selection
        i_ub = j + 1
        new_samples = sample_from_sun(d1_array[:i_ub], s_array[:i_ub], d2_array[:i_ub], mc_samples)
        avail_items = np.setdiff1d(np.arange(m), item_selections[:i_ub])
        p_sampled = norm.cdf(
            alphas[avail_items, :] @ new_samples.T + intercepts[avail_items].reshape(-1, 1))  # j * mc_samples
        pred_var = np.var(p_sampled, axis=1)
        temp_item = avail_items[np.argmax(pred_var)]
    return item_selections

def initialize_predictions(alphas, intercepts, mc_samples):
    m, k = alphas.shape
    thetas = np.random.multivariate_normal(np.zeros(k), np.identity(k), mc_samples)
    linear_term = thetas @ alphas.T + intercepts
    pred = norm.cdf(linear_term)  # s by j
    preds = np.zeros((m, 11))
    preds[:, 0] = np.mean(pred, axis=0)
    preds[:, 1] = np.var(pred, axis=0)
    for i, q in enumerate(np.arange(0.1, 1, 0.1)):
        preds[:, (i + 2)] = np.quantile(pred, q=q, axis=0)
    return preds

def initialize_q_networks(alphas):
    m, k = alphas.shape
    input_dim = k + 2
    input_stats_dim = int(k * 4 + (1 + k) * k / 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = OnlineQNetworkV4(
                input_dim=input_dim,
                input_stats_dim=input_stats_dim,
                num_items=m,
                hidden_dim_phi1=256,
            ).to(device)
    return network


def test_q_learning_speed(theta, test_params, mc_samples, network_dir, max_items):
    alphas, intercepts = test_params["alphas"], test_params["intercepts"]
    m, k = alphas.shape
    max_items = max_items+1
    d1_array = np.zeros((max_items, k))
    d2_array = np.zeros(max_items)
    s_array = np.zeros(max_items)
    cov_mat = np.identity(k)
    dist_stats = {0.25: np.ones(k)*(-0.67448), 0.5: np.zeros(k), 0.75: np.ones(k)*0.67448, "mean": np.zeros(k),
                  "cov": cov_mat[np.triu_indices_from(cov_mat)]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds = initialize_predictions(alphas, intercepts, mc_samples)
    network = initialize_q_networks(alphas)
    network.load_state_dict(torch.load(network_dir))
    item_selections = np.zeros(max_items).astype(int)
    for j in tqdm(range(1, max_items)):
        if j > 1:
            new_samples = sample_from_sun(d1_array[1:j], s_array[1:j], d2_array[1:j], mc_samples)
        d1_temp = d1_array[:j]  # (n, h, k)
        d2_temp = d2_array[:j]  # (n,h)
        s_array_temp = s_array[:j]
        x = np.concatenate(
            [np.concatenate((d1_temp[h], np.array([d2_temp[h], s_array_temp[h]]))) for h in range(j)])
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        x_tensor.to(device)
        if j > 1:
            q1 = np.quantile(new_samples, 0.25, axis=0)
            q2 = np.quantile(new_samples, 0.5, axis=0)
            q3 = np.quantile(new_samples, 0.75, axis=0)
            mean = np.mean(new_samples, axis=0)
            cov_mat = np.cov(new_samples.T)
            cov = cov_mat[np.triu_indices_from(cov_mat)]
            linear_term = new_samples @ alphas.T + intercepts
            pred = norm.cdf(linear_term)  # s by j
            x_preds = np.zeros((m, 11))
            x_preds[:, 0] = np.mean(pred, axis=0)
            x_preds[:, 1] = np.var(pred, axis=0)
            for l, q in enumerate(np.arange(0.1, 1, 0.1)):
                x_preds[:, (l + 2)] = np.quantile(pred, q=q, axis=0)
        else:
            q1 = dist_stats[0.25]
            q2 = dist_stats[0.5]
            q3 = dist_stats[0.75]
            mean = np.zeros(k)
            cov = dist_stats["cov"]
            x_preds = preds

        x_stats = np.concatenate((q1, q2, q3, mean, cov))
        x_stats_tensor = torch.tensor(x_stats, dtype=torch.float32).unsqueeze(0)
        x_stats_tensor.to(device)
        x_preds_tensor = torch.tensor(x_preds, dtype=torch.float32).unsqueeze(0)
        mask = np.ones(m)
        mask[item_selections[:(j-1)]] = 0
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        mask_tensor.to(device)
        network.eval()
        with torch.no_grad():
            q_values = network(
                x_tensor,
                x_stats_tensor,
                x_preds_tensor,
                mask_tensor
            )
        temp_item = torch.argmax(q_values, dim=1)
        item_selections[j-1] = temp_item.item()
        # update posterior params
        temp_alpha = alphas[temp_item]
        temp_intercept = intercepts[temp_item]
        linear_term = np.sum(alphas * theta) + temp_intercept
        prob = norm.cdf(linear_term)
        response = (np.random.uniform(0, 1, 1) < prob).astype(int)
        d1_array[j] = (2 * response - 1) * temp_alpha
        d2_array[j] = (2 * response - 1) * temp_intercept
        s_array[j] = (np.sum(d1_array[j] * d1_array[j]) + 1) ** 0.5
    return item_selections


if __name__ == "__main__":
    # data setup
    arguments = parse_args()
    data_input_dir = pathlib.Path(arguments["sim_data_dir"], arguments["input_data_name"])
    with open(data_input_dir, 'rb') as handle:
        test_params = pickle.load(handle)
    # testing speeds
    np.random.seed(arguments["seed"])
    theta = np.random.multivariate_normal(np.zeros(5), np.identity(5), 1)
    print("Testing kl eap speed")
    item_selections= test_kl_eap_speed(theta, test_params, arguments["mc_samples"], arguments["max_items"])
    print(item_selections)
    print("Testing kl pos speed")
    item_selections= test_kl_pos_speed(theta, test_params, arguments["mc_samples"], arguments["max_items"])
    print(item_selections)
    print("Testing mi sir speed")
    item_selections= test_mi_sir_speed(theta, test_params, arguments["mc_samples"], arguments["sir_large_samples"], arguments["max_items"])
    print(item_selections)
    print("Testing prediction ve speed")
    item_selections= test_prediction_ve_speed(theta, test_params, arguments["mc_samples"], arguments["max_items"])
    print(item_selections)
    print("Testing Q learning Speed")
    filename = "retrain_retrainv1_NN_x09_s01_mixed_grid_500_200_5_4_150000_first3-0-1_detailed_v4_1.5e-05_128_30000_1000_0.16/retrain_fn_24000.pt_fn_11000.pt"
    nn_input_dir = pathlib.Path(arguments["q_learning_models_dir"]).joinpath("online-Q-network-v4-5-factor-standarized-first3").joinpath(filename)
    item_selections= test_q_learning_speed(theta, test_params, arguments["mc_samples"], nn_input_dir, arguments["max_items"])
    print(item_selections)

