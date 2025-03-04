import typing
import numpy as np
from .test_taker import TestTakers
from tqdm import tqdm
from .deep_q_network import QNetwork as Model
from .deep_q_network import OnlineQNetwork, OnlineQNetworkV2, OnlineQNetworkV3, OnlineQNetworkV4, OnlineQNetworkV5
from .prediction_network import OnlinePredNNV2, OnlinePredNNV1, OnlinePredNNV3
import torch
import os
from scipy.stats import norm

class DeepQCAT:

    def __init__(
            self,
            true_thetas: np.ndarray,
            test_bank: typing.Dict[str, np.ndarray],
            nn_model_dir: typing.Union[str, os.PathLike],
            max_items: int,
            mc_samples: int,
            plugin_response: bool = False,
            true_response: np.ndarray = None,
            num_workers: int = 8,
            random_state: int = 42,
    ):
        """
        CAT with Deep Q Learning
        """
        self.seed = random_state
        self.true_thetas = true_thetas
        self.n = true_thetas.shape[0]
        self.k = true_thetas.shape[1]
        self.alphas = test_bank["alphas"]
        self.intercepts = test_bank["intercepts"]
        self.max_items = max_items
        self.mc_samples = mc_samples
        self.num_workers = num_workers
        self.rng = np.random.default_rng(random_state)
        self.m = self.alphas.shape[0]
        self.theta_updates = np.zeros((self.max_items, self.n, self.k))
        self.item_selections = np.zeros((self.n, self.max_items))
        self.plugin_response = plugin_response
        self.response = true_response
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = self.k + 2
        input_stats_dim = int(self.k * 4 + (1 + self.k) * self.k / 2)
        self.nn = Model(
            input_dim=input_dim,
            input_stats_dim=input_stats_dim,
            num_items= self.m,
            hidden_dim_phi1=256,
            hidden_dim_stats=256
        ).to(self.device)
        self.nn.load_state_dict(torch.load(nn_model_dir))
        self.tts = TestTakers(thetas=self.true_thetas, max_item=self.max_items, num_workers=self.num_workers,
                              random_state=self.seed)

    def simulate(self):
        """For each test taker, simulate their responses to selected items and record theta_updates"""
        for j in tqdm(range(1, self.max_items+1)):
            new_items = self.select_items(j)
            if self.plugin_response:
                true_responses = self.response[np.arange(self.response.shape[0]), new_items]
                self.tts.answer_true_responses(self.alphas[new_items], self.intercepts[new_items], new_items,
                                      response_vec =true_responses.astype(int))
            else:
                self.tts.answer_items(self.alphas[new_items], self.intercepts[new_items], new_items)
        self.item_selections = self.tts.item_ids

    def select_items(self, item_idx):
        """Select Next Items Depending on Given Selection Criterion for each test taker"""
        result = np.zeros(self.n, dtype=int)
        if item_idx > 1:
            new_samples = self.tts.get_factor_samples(self.mc_samples)  # n by num_sir_samples by k
        d1_tensor = self.tts.d1_tensor[:, :item_idx, :]  # (n, h, k)
        d2_array = self.tts.d2_array[:, :item_idx] # (n,h)
        s_array = self.tts.s_array[:, :item_idx]
        for i in range(self.n):
            d1_temp = d1_tensor[i] # h by k
            d2_temp = d2_array[i]  # h
            s_array_temp = s_array[i]  # h
            x = np.concatenate(
                [np.concatenate((d1_temp[h], np.array([d2_temp[h], s_array_temp[h]]))) for h in range(item_idx)])
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            x_tensor.to(self.device)
            if item_idx > 1:
                q1 =  np.quantile(new_samples[i], 0.25, axis=0)
                q2 = np.quantile(new_samples[i], 0.5, axis=0)
                q3 = np.quantile(new_samples[i], 0.75, axis=0)
                mean = np.mean(new_samples[i], axis=0)
                cov_mat = np.cov(new_samples[i].T)
                cov = cov_mat[np.triu_indices_from(cov_mat)]
            else:
                q1 = self.tts.dist_stats[0.25][i, 0, :]
                q2 = self.tts.dist_stats[0.5][i, 0, :]
                q3 = self.tts.dist_stats[0.75][i, 0, :]
                mean = np.zeros(self.k)
                cov_mat = np.identity(self.k)
                cov = cov_mat[np.triu_indices_from(cov_mat)]

            x_stats = np.concatenate((q1, q2, q3, mean, cov))
            x_stats_tensor = torch.tensor(x_stats, dtype=torch.float32).unsqueeze(0)
            x_stats_tensor.to(self.device)
            mask = np.ones(self.m)
            mask[self.tts.item_ids[i, :(item_idx-1)]] = 0
            mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            mask_tensor.to(self.device)
            self.nn.eval()
            with torch.no_grad():
                q_values = self.nn(
                    torch.tensor(x, dtype=torch.float32).unsqueeze(0),
                    torch.tensor(x_stats, dtype=torch.float32).unsqueeze(0),
                    torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
                )
            predicted_item= torch.argmax(q_values , dim=1)
            result[i] = predicted_item

        return result


############################################################################################

class OnlineDeepQCAT:

    def __init__(
            self,
            true_thetas: np.ndarray,
            test_bank: typing.Dict[str, np.ndarray],
            nn_model_dir: typing.Union[str, os.PathLike],
            max_items: int,
            mc_samples: int,
            network_version: str,
            pred_type: str,
            plugin_response: bool = False,
            true_response: np.ndarray = None,
            pred_only: bool = False,
            num_workers: int = 8,
            random_state: int = 42,
    ):
        """
        CAT with Deep Q Learning
        """
        self.seed = random_state
        self.true_thetas = true_thetas
        self.n = true_thetas.shape[0]
        self.k = true_thetas.shape[1]
        self.alphas = test_bank["alphas"]
        self.intercepts = test_bank["intercepts"]
        self.max_items = max_items
        self.mc_samples = mc_samples
        self.num_workers = num_workers
        self.pred_type = pred_type
        self.rng = np.random.default_rng(random_state)
        self.m = self.alphas.shape[0]
        self.theta_updates = np.zeros((self.max_items, self.n, self.k))
        self.item_selections = np.zeros((self.n, self.max_items))
        self.plugin_response = plugin_response
        self.response = true_response
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pred_only = pred_only
        input_dim = self.k + 2
        input_stats_dim = int(self.k * 4 + (1 + self.k) * self.k / 2)
        input_pred_dim = 11
        if network_version == "v1":
            self.nn = OnlineQNetwork(
                input_dim=input_dim,
                input_stats_dim=input_stats_dim,
                num_items=self.m,
                hidden_dim_phi1=256,
            ).to(self.device)
        elif network_version == "v2":
            self.nn = OnlineQNetworkV2(
                input_dim=input_dim,
                input_stats_dim=input_stats_dim,
                num_items=self.m,
                hidden_dim_phi1=256,
            ).to(self.device)
        elif network_version == "v3":
            self.nn = OnlineQNetworkV3(
                input_dim=input_dim,
                input_stats_dim=input_stats_dim,
                num_items= self.m,
                hidden_dim_phi1=256,
            ).to(self.device)
        elif network_version == "v4":
            self.nn = OnlineQNetworkV4(
                input_dim=input_dim,
                input_stats_dim=input_stats_dim,
                num_items=self.m,
                hidden_dim_phi1=256,
            ).to(self.device)
        elif network_version == "v5":
            self.nn = OnlineQNetworkV5(
                input_dim=input_dim,
                input_stats_dim=input_stats_dim,
                num_items=self.m,
                hidden_dim_phi1=256,
            ).to(self.device)
        elif network_version == "pred_v1":
            self.nn = OnlinePredNNV1(
                pred_stats_dim=input_pred_dim,
                num_items=self.m,
                hidden_dim=1024
            ).to(self.device)
        elif network_version == "pred_v2":
            self.nn = OnlinePredNNV2(
                pred_stats_dim=input_pred_dim,
                num_items=self.m,
                num_heads=8,
                num_layers=2
            ).to(self.device)
        else:
            self.nn = OnlinePredNNV3(
                    pred_stats_dim=input_pred_dim,
                    hidden_dim= 64,
                    output_dim= 1
                ).to(self.device)

        self.nn.load_state_dict(torch.load(nn_model_dir))
        self.tts = TestTakers(thetas=self.true_thetas, max_item=self.max_items, num_workers=self.num_workers,
                              random_state=self.seed)
        if pred_type == "basic":
            self.preds = np.zeros(self.m*2)
        else:
            self.preds = np.zeros((self.m, 11)) # mean variance: 10, 20, ...,90, actually no need to store it for all
        self.initialize_predictions()

    def initialize_predictions(self):
        """Initialize Predictions for Multivariate Normal"""
        thetas = self.rng.multivariate_normal(np.zeros(self.k), np.identity(self.k), self.mc_samples)
        linear_term = thetas @ self.alphas.T + self.intercepts
        pred = norm.cdf(linear_term)  # s by j
        if self.pred_type == "basic":
            pred_mean = np.mean(pred, axis=0)
            pred_var = np.var(pred, axis=0)
            self.preds[:self.m] = pred_var
            self.preds[-self.m:] = pred_mean
        else:
            self.preds[:, 0] = np.mean(pred, axis=0)
            self.preds[:, 1] = np.var(pred, axis=0)
            for i, q in enumerate(np.arange(0.1, 1, 0.1)):
                self.preds[:, (i + 2)] = np.quantile(pred, q=q, axis=0)

    def simulate(self):
        """For each test taker, simulate their responses to selected items and record theta_updates"""
        for j in tqdm(range(1, self.max_items+1)):
            if self.pred_only:
                new_items = self.select_pred_only_items(j)
            else:
                new_items = self.select_items(j)
            if self.plugin_response:
                true_responses = self.response[np.arange(self.response.shape[0]), new_items]
                self.tts.answer_true_responses(self.alphas[new_items], self.intercepts[new_items], new_items,
                                               response_vec=true_responses.astype(int))
            else:
                self.tts.answer_items(self.alphas[new_items], self.intercepts[new_items], new_items)
        self.item_selections = self.tts.item_ids

    def select_items(self, item_idx):
        """Select Next Items Depending on Given Selection Criterion for each test taker"""
        result = np.zeros(self.n, dtype=int)
        if item_idx > 1:
            new_samples = self.tts.get_factor_samples(self.mc_samples)  # n by num_sir_samples by k
        d1_tensor = self.tts.d1_tensor[:, :item_idx, :]  # (n, h, k)
        d2_array = self.tts.d2_array[:, :item_idx] # (n,h)
        s_array = self.tts.s_array[:, :item_idx]
        for i in range(self.n):
            d1_temp = d1_tensor[i] # h by k
            d2_temp = d2_array[i]  # h
            s_array_temp = s_array[i]  # h
            x = np.concatenate(
                [np.concatenate((d1_temp[h], np.array([d2_temp[h], s_array_temp[h]]))) for h in range(item_idx)])
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            x_tensor.to(self.device)
            if item_idx > 1:
                q1 = np.quantile(new_samples[i], 0.25, axis=0)
                q2 = np.quantile(new_samples[i], 0.5, axis=0)
                q3 = np.quantile(new_samples[i], 0.75, axis=0)
                mean = np.mean(new_samples[i], axis=0)
                cov_mat = np.cov(new_samples[i].T)
                cov = cov_mat[np.triu_indices_from(cov_mat)]
                linear_term = new_samples[i] @ self.alphas.T + self.intercepts
                pred = norm.cdf(linear_term)  # s by j
                if self.pred_type == "basic":
                    pred_mean = np.mean(pred, axis=0)
                    pred_var = np.var(pred, axis=0)
                    x_preds = np.zeros(self.m*2)
                    x_preds[:self.m] = pred_var
                    x_preds[-self.m:] = pred_mean
                else:
                    x_preds = np.zeros((self.m, 11))
                    x_preds[:, 0] = np.mean(pred, axis=0)
                    x_preds[:, 1] = np.var(pred, axis=0)
                    for j, q in enumerate(np.arange(0.1, 1, 0.1)):
                        x_preds[:, (j + 2)] = np.quantile(pred, q=q, axis=0)

            else:
                q1 = self.tts.dist_stats[0.25][i, 0, :]
                q2 = self.tts.dist_stats[0.5][i, 0, :]
                q3 = self.tts.dist_stats[0.75][i, 0, :]
                mean = np.zeros(self.k)
                cov_mat = np.identity(self.k)
                cov = cov_mat[np.triu_indices_from(cov_mat)]
                x_preds = self.preds

            x_stats = np.concatenate((q1, q2, q3, mean, cov))
            x_stats_tensor = torch.tensor(x_stats, dtype=torch.float32).unsqueeze(0)
            x_stats_tensor.to(self.device)

            if self.pred_type == "basic":
                x_preds[self.tts.item_ids[i, :(item_idx-1)]] = 0
                x_preds_tensor = torch.tensor(x_preds, dtype=torch.float32).unsqueeze(0)
            else:
                x_preds_tensor = torch.tensor(x_preds, dtype=torch.float32).unsqueeze(0)

            x_preds_tensor.to(self.device)

            mask = np.ones(self.m)
            mask[self.tts.item_ids[i, :(item_idx-1)]] = 0
            mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            mask_tensor.to(self.device)
            self.nn.eval()
            with torch.no_grad():
                q_values = self.nn(
                    x_tensor,
                    x_stats_tensor,
                    x_preds_tensor,
                    mask_tensor
                )
            predicted_item= torch.argmax(q_values , dim=1)
            result[i] = predicted_item

        return result

    def select_pred_only_items(self, item_idx):
        """Select Next Items Depending on Given Selection Criterion for each test taker"""
        result = np.zeros(self.n, dtype=int)
        if item_idx > 1:
            new_samples = self.tts.get_factor_samples(self.mc_samples)  # n by num_sir_samples by k
        for i in range(self.n):
            if item_idx > 1:
                linear_term = new_samples[i] @ self.alphas.T + self.intercepts
                pred = norm.cdf(linear_term)  # s by j
                x_preds = np.zeros((self.m, 11))
                x_preds[:, 0] = np.mean(pred, axis=0)
                x_preds[:, 1] = np.var(pred, axis=0)
                for j, q in enumerate(np.arange(0.1, 1, 0.1)):
                    x_preds[:, (j + 2)] = np.quantile(pred, q=q, axis=0)

            else:
                x_preds = self.preds

            x_preds_tensor = torch.tensor(x_preds, dtype=torch.float32).unsqueeze(0)
            x_preds_tensor.to(self.device)

            mask = np.ones(self.m)
            mask[self.tts.item_ids[i, :(item_idx-1)]] = 0
            mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            mask_tensor.to(self.device)
            self.nn.eval()
            with torch.no_grad():
                q_values = self.nn(
                    x_preds_tensor,
                    mask_tensor
                )
            predicted_item= torch.argmax(q_values , dim=1)
            result[i] = predicted_item
        return result

###############################################

class PluginOnlineDeepQCATV4:

    def __init__(
            self,
            true_thetas: np.ndarray,
            test_bank: typing.Dict[str, np.ndarray],
            nn_model_dir: typing.Union[str, os.PathLike],
            max_items: int,
            mc_samples: int,
            gamma: float,
            plugin_response: bool = False,
            true_response: np.ndarray = None,
            num_workers: int = 8,
            random_state: int = 42,
    ):
        """
        CAT with Deep Q Learning
        """
        self.seed = random_state
        self.true_thetas = true_thetas
        self.n = true_thetas.shape[0]
        self.k = true_thetas.shape[1]
        self.alphas = test_bank["alphas"]
        self.intercepts = test_bank["intercepts"]
        self.max_items = max_items
        self.mc_samples = mc_samples
        self.num_workers = num_workers
        self.rng = np.random.default_rng(random_state)
        self.m = self.alphas.shape[0]
        self.theta_updates = np.zeros((self.max_items, self.n, self.k))
        self.item_selections = np.zeros((self.n, self.max_items))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.plugin_response = plugin_response
        self.response = true_response
        input_dim = self.k + 2
        input_stats_dim = int(self.k * 4 + (1 + self.k) * self.k / 2)
        self.nn = OnlineQNetworkV4(
            input_dim=input_dim,
            input_stats_dim=input_stats_dim,
                num_items=self.m,
                hidden_dim_phi1=256,
            ).to(self.device)
        self.nn.load_state_dict(torch.load(nn_model_dir))
        self.tts = TestTakers(thetas=self.true_thetas, max_item=self.max_items, num_workers=self.num_workers,
                              random_state=self.seed)
        self.gamma = gamma

    def simulate(self):
        """For each test taker, simulate their responses to selected items and record theta_updates"""
        for j in tqdm(range(1, self.max_items+1)):
            new_items = self.select_items(j)
            if self.plugin_response:
                true_responses = self.response[np.arange(self.response.shape[0]), new_items]
                self.tts.answer_true_responses(self.alphas[new_items], self.intercepts[new_items], new_items,
                                               response_vec=true_responses.astype(int))

            else:
                self.tts.answer_items(self.alphas[new_items], self.intercepts[new_items], new_items)
        self.item_selections = self.tts.item_ids

    def select_items(self, item_idx):
        """Select Next Items Depending on Given Selection Criterion for each test taker"""
        result = np.zeros(self.n, dtype=int)
        if item_idx > 1:
            new_samples = self.tts.get_factor_samples(self.mc_samples)  # n by num_sir_samples by k
        else:
            new_samples =  self.rng.multivariate_normal(np.zeros(self.k), np.identity(self.k), self.mc_samples)
        d1_tensor = self.tts.d1_tensor[:, :item_idx, :]  # (n, h, k)
        d2_array = self.tts.d2_array[:, :item_idx] # (n,h)
        s_array = self.tts.s_array[:, :item_idx]
        for i in range(self.n):
            print(i)
            if item_idx > 1:
                pos_samples = new_samples[i]
            else:
                pos_samples = new_samples
            # get future samples
            avail_items = np.setdiff1d(np.arange(self.m), self.tts.item_ids[i, :item_idx])
            cum_rewards = np.zeros(avail_items.shape[0])
            for j, a in enumerate(avail_items):
                # future posteriors
                prob1 = norm.cdf(pos_samples @ self.alphas[a] + self.intercepts[a])  # sir_large_samples
                pred_hat = np.mean(prob1)
                threshold = 1e-10
                prob1 = np.where(prob1 > (1 - threshold), 1 - threshold, prob1)
                prob1 = np.where(prob1 < threshold, threshold, prob1)
                prob0 = 1 - prob1
                weight1 = prob1 / np.sum(prob1)  # j * sir_large_samples
                weight0 = prob0 / np.sum(prob0)
                indices0 = np.random.choice(a=self.mc_samples, size=self.mc_samples, p=weight0)
                indices1 = np.random.choice(a=self.mc_samples, size=self.mc_samples, p=weight1)
                future_pos_1 = pos_samples[indices1]
                future_pos_0 = pos_samples[indices0]
                # compute mi
                p_sampled0 = 1 - norm.cdf(future_pos_0 @ self.alphas[a] + self.intercepts[a])
                p_sampled1 = norm.cdf(future_pos_1 @ self.alphas[a] + self.intercepts[a])
                mi_term1 = pred_hat * (np.mean(np.log(p_sampled1)) - np.log(pred_hat))
                mi_term2 = (1 - pred_hat) * (np.mean(np.log(p_sampled0)) - np.log(1 - pred_hat))
                mi = mi_term1 + mi_term2
                # plug in Q network
                for y in [0,1]:
                    # next parameters
                    d1_temp = np.vstack([d1_tensor[i], (2*y-1)*self.alphas[a]])  # h by k
                    d2_temp = np.append(d2_array[i], (2*y-1)* self.intercepts[a])
                    s_array_temp = np.append(s_array[i], (np.sum(np.square(self.alphas[a]))+1)**0.5)
                    x = np.concatenate(
                        [np.concatenate((d1_temp[h], np.array([d2_temp[h], s_array_temp[h]]))) for h in
                         range(item_idx+1)])
                    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                    x_tensor.to(self.device)
                    # next predictions
                    if y == 0:
                        future_samples = future_pos_0
                    else:
                        future_samples = future_pos_1
                    q1 = np.quantile(future_samples, 0.25, axis=0)
                    q2 = np.quantile(future_samples, 0.5, axis=0)
                    q3 = np.quantile(future_samples, 0.75, axis=0)
                    mean = np.mean(future_samples, axis=0)
                    cov_mat = np.cov(future_samples.T)
                    cov = cov_mat[np.triu_indices_from(cov_mat)]
                    linear_term = future_samples @ self.alphas.T + self.intercepts
                    pred = norm.cdf(linear_term)  # s by j
                    x_preds = np.zeros((self.m, 11))
                    x_preds[:, 0] = np.mean(pred, axis=0)
                    x_preds[:, 1] = np.var(pred, axis=0)
                    for l, q in enumerate(np.arange(0.1, 1, 0.1)):
                        x_preds[:, (l + 2)] = np.quantile(pred, q=q, axis=0)

                    x_stats = np.concatenate((q1, q2, q3, mean, cov))
                    x_stats_tensor = torch.tensor(x_stats, dtype=torch.float32).unsqueeze(0)
                    x_stats_tensor.to(self.device)
                    x_preds_tensor = torch.tensor(x_preds, dtype=torch.float32).unsqueeze(0)
                    x_preds_tensor.to(self.device)

                    mask = np.ones(self.m)
                    mask[self.tts.item_ids[i, :(item_idx - 1)]] = 0
                    mask[a] = 0
                    mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
                    mask_tensor.to(self.device)
                    self.nn.eval()
                    with torch.no_grad():
                        q_values = self.nn(
                            x_tensor,
                            x_stats_tensor,
                            x_preds_tensor,
                            mask_tensor
                        )
                    if y == 0:
                        next_q_val = self.gamma * (1-pred_hat) * torch.max(q_values).item()
                    else:
                        next_q_val = self.gamma * pred_hat * torch.max(q_values).item()
                    mi += next_q_val
                cum_rewards[j] = mi
            result[i] = avail_items[np.argmax(cum_rewards)]
        return result




