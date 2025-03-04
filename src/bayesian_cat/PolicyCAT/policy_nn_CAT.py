import typing
import numpy as np
from .test_taker import TestTakers
from tqdm import tqdm
from .NN_models import OnlinePolicyNetwork as Model
import torch
import os


class PolicyNNCAT:

    def __init__(
            self,
            true_thetas: np.ndarray,
            test_bank: typing.Dict[str, np.ndarray],
            nn_model_dir: typing.Union[str, os.PathLike],
            max_items: int,
            mc_samples: int,
            starting_item: typing.Optional[int] = None,
            num_workers: int = 8,
            random_state: int = 42,
    ):
        """

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
        input_dim = self.k + 2
        input_stats_dim = self.k * 4 + self.k ** 2
        self.nn = Model(input_dim, input_stats_dim, self.m, hidden_dim=256).to(self.device)
        self.nn.load_state_dict(torch.load(nn_model_dir))
        if starting_item:
            self.start = starting_item
        else:
            self.start = self.pick_starting_item()
        self.tts = TestTakers(thetas=self.true_thetas, max_item=self.max_items, num_workers=self.num_workers,
                              random_state=self.seed)

    def pick_starting_item(self):
        """Pick the starting item from the test bank

        Currently, consider picking item with around median intercept, and relatively high magnitude of alpha parameters
        """
        icpt_40 = np.quantile(self.intercepts, 0.4)
        icpt_60 = np.quantile(self.intercepts, 0.6)
        indices = np.where((self.intercepts >= icpt_40) & (self.intercepts <= icpt_60))[0]
        abs_sum = np.sum(np.abs(self.alphas[indices]), axis=1)
        upper_quantile_value = np.quantile(abs_sum, 0.75)
        arg_max_upper_quantile = np.argmin(np.abs(abs_sum - upper_quantile_value))
        return indices[arg_max_upper_quantile]

    def simulate(self):
        """For each test taker, simulate their responses to selected items and record theta_updates"""
        start_items = np.array([self.start]*self.n)
        self.tts.answer_items(self.alphas[start_items], self.intercepts[start_items], start_items)
        for j in tqdm(range(1, self.max_items)):
            new_items = self.select_items(j)
            self.tts.answer_items(self.alphas[new_items], self.intercepts[new_items], new_items)
        self.theta_updates = self.tts.pos_mean
        self.item_selections = self.tts.item_ids

    def select_items(self, item_idx):
        """Select Next Items Depending on Given Selection Criterion for each test taker"""
        result = np.zeros(self.n, dtype=int)
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
            q1 =  np.quantile(new_samples[i], 0.25, axis=0)
            q2 = np.quantile(new_samples[i], 0.5, axis=0)
            q3 = np.quantile(new_samples[i], 0.75, axis=0)
            mean = np.mean(new_samples[i], axis=0)
            cov = np.cov(new_samples[i].T).flatten()
            x_stats = np.concatenate((q1, q2, q3, mean, cov))
            x_stats_tensor = torch.tensor(x_stats, dtype=torch.float32).unsqueeze(0)
            x_stats_tensor.to(self.device)
            mask = np.ones(self.m)
            mask[self.tts.item_ids[i, :item_idx]] = 0
            mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            mask_tensor.to(self.device)
            self.nn.eval()
            with torch.no_grad():
                logits = self.nn(x_tensor, x_stats_tensor, mask_tensor)
                predicted_item = torch.argmax(logits, dim=1)
            predicted_item= predicted_item.cpu().numpy()
            result[i] = int(predicted_item[0].item())

        return result
