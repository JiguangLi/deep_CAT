import typing
import numpy as np
from .test_taker import BayesianQTestTakers
from tqdm import tqdm
import torch
import os
from scipy.stats import norm
from .deep_q_network import BayesianOnlineQNetworkV4


class BayesianOnlineDeepQCAT:

    def __init__(
            self,
            true_thetas: np.ndarray,
            test_bank: typing.Dict[str, np.ndarray],
            nn_model_dir: typing.Union[str, os.PathLike],
            max_items: int,
            mc_samples: int,
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
        self.pos_num_samples,self.m = self.alphas.shape[0] ,self.alphas.shape[1]
        self.item_selections = np.zeros((self.n, self.max_items))
        self.plugin_response = plugin_response
        self.response = true_response
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pred_only = pred_only
        input_dim = self.k + 2
        input_stats_dim = int(self.k * 4 + (1 + self.k) * self.k / 2)
        self.nn = BayesianOnlineQNetworkV4(
            input_dim=input_dim,
            input_stats_dim=input_stats_dim,
            num_items=self.m,
            hidden_dim_phi1=256,
        ).to(self.device)
        self.nn.load_state_dict(torch.load(nn_model_dir))
        self.tts = BayesianQTestTakers(thetas=self.true_thetas,
                                       pos_num_samples= self.pos_num_samples,
                                       max_item=self.max_items,
                                       num_workers=self.num_workers,
                                       random_state=self.seed)

        self.preds = np.zeros((self.m, 11)) # mean variance: 10, 20, ...,90, actually no need to store it for all
        self.initialize_predictions()

    def initialize_predictions(self):
        """Initialize Predictions for Multivariate Normal"""
        thetas = self.rng.multivariate_normal(np.zeros(self.k), np.identity(self.k),
                                              size=self.pos_num_samples * self.mc_samples)
        thetas = thetas.reshape(self.pos_num_samples, self.mc_samples, self.k)
        p_sampled = np.einsum('pak,pmk->pam', self.alphas, thetas)  # pos_samples by m by mc_samples
        p_sampled += self.intercepts[:, :, np.newaxis]
        p_sampled = np.transpose(p_sampled, axes=(1, 0, 2))  # m by p_samples by mc_samples
        p_sampled = p_sampled.reshape(self.m, self.mc_samples * self.pos_num_samples)  # a by (p*m)
        pred = norm.cdf(p_sampled).T  # (p*mc) * j
        self.preds[:, 0] = np.mean(pred, axis=0)
        self.preds[:, 1] = np.var(pred, axis=0)
        for i, q in enumerate(np.arange(0.1, 1, 0.1)):
            self.preds[:, (i + 2)] = np.quantile(pred, q=q, axis=0)

    def simulate(self):
        """For each test taker, simulate their responses to selected items and record theta_updates"""
        for j in tqdm(range(1, self.max_items+1)):
            new_items = self.select_items(j)
            true_responses = self.response[np.arange(self.response.shape[0]), new_items]
            self.tts.answer_true_responses(self.alphas[new_items], self.intercepts[new_items], new_items,
                                           response_vec=true_responses.astype(int))
        self.item_selections = self.tts.item_ids

    def select_items(self, item_idx):
        """Select Next Items Depending on Given Selection Criterion for each test taker"""
        result = np.zeros(self.n, dtype=int)
        if item_idx > 1:
            new_samples = self.tts.get_factor_samples(self.mc_samples)  # n by pos_samples* num_sir_samples by k
        d1_tensor = self.tts.d1_tensor[:, :, :item_idx, :]  # (n,p , h, k)
        d2_array = self.tts.d2_array[:, :, :item_idx] # (n,p, h)
        s_array = self.tts.s_array[:, :, :item_idx]
        for i in range(self.n):
            d1_temp = d1_tensor[i] # p by h by k
            d2_temp = d2_array[i]  # p by h
            s_array_temp = s_array[i]  # p by h
            x = np.concatenate(
                [np.concatenate((np.mean(d1_temp[:, h, :], axis=0),
                                 np.array([np.mean(d2_temp[:, h]),
                                           np.mean(s_array_temp[:, h])]))) for h in range(item_idx)])
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            x_tensor.to(self.device)
            if item_idx > 1:
                q1 = np.quantile(new_samples[i], 0.25, axis=(0,1))
                q2 = np.quantile(new_samples[i], 0.5, axis=(0,1))
                q3 = np.quantile(new_samples[i], 0.75, axis=(0,1))
                mean = np.mean(new_samples[i], axis=(0,1))
                cov_mat = np.cov(new_samples[i].reshape(-1, self.k).T)
                cov = cov_mat[np.triu_indices_from(cov_mat)]
                p_sampled = np.einsum('pak,pmk->pam', self.alphas, new_samples[i])  # pos_samples by m by mc_samples
                p_sampled += self.intercepts[:, :, np.newaxis]
                p_sampled = np.transpose(p_sampled, axes=(1, 0, 2))  # m by p_samples by mc_samples
                p_sampled = p_sampled.reshape(self.m, self.mc_samples * self.pos_num_samples)  # a by (p*m)
                pred = norm.cdf(p_sampled).T  # (p*mc) * j
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
