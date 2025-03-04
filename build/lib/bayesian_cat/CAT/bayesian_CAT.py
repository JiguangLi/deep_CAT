import typing
import numpy as np
from .test_taker import TestTakers
from tqdm import tqdm
from scipy.stats import norm


class BayesianCAT:

    def __init__(
            self,
            true_thetas: np.ndarray,
            test_bank: typing.Dict[str, np.ndarray],
            selection_criterion: str,
            starting_item: int,
            max_items: int,
            mc_samples: int,
            num_workers: int = 8,
            random_state: int = 42,
    ):
        """
        Create a Bayesian Computerized Testing Object for Simulation

            Args:
                true_thetas: Shape: (n, k), latent factors of new test takers
                test_bank: dictionary with loading matrix "alphas", and "intercepts" for the items in the test bank
                selection_criterion: kl_eap, kl_pos, or mi
                starting_item: first item to start
                max_items: maximum numbers to test, currently no stopping criterions for the simulation
                mc_samples: number of mc samples
                num_workers: number of cores for multiprocessing
                random_state: seed
            Raises:
                ValueError: if given inputs do not meet object expectations.
        """
        self.rng = np.random.default_rng(random_state)
        self.true_thetas = true_thetas
        self.n = true_thetas.shape[0]
        self.k = true_thetas.shape[1]
        self.alphas = test_bank["alphas"]
        self.intercepts = test_bank["intercepts"]
        self.sc = selection_criterion
        self.start = starting_item
        self.max_items = max_items
        self.mc_samples = mc_samples
        self.num_workers = num_workers
        self.m = self.alphas.shape[0]
        self.theta_updates = np.zeros((self.max_items, self.n, self.k))
        self.item_selections = np.zeros((self.n, self.max_items))

    def simulate(self):
        """For each test taker, simulate their responses to selected items and record theta_updates"""
        tts = TestTakers(thetas=self.true_thetas, max_item=self.max_items, num_workers=self.num_workers)
        start_items = np.array([self.start]*self.n)
        tts.answer_items(self.alphas[start_items], self.intercepts[start_items], start_items)
        tts.get_pos_mean(0, self.mc_samples)
        for j in tqdm(range(1, self.max_items)):
            new_items = self.select_items(j, tts)
            tts.answer_items(self.alphas[new_items], self.intercepts[new_items], new_items)
            tts.get_pos_mean(j, self.mc_samples)
        self.theta_updates = tts.pos_mean
        self.item_selections = tts.item_ids

    def select_kl_eap(self, item_idx: int, tts: TestTakers):
        """Select Items Based on Posterior Expected KL Information"""
        result = np.zeros(self.n)
        pos_mean = tts.pos_mean[item_idx-1] # n by k
        new_samples = tts.get_factor_samples(self.mc_samples) # n by num_samples by k
        for i in range(self.n):
            theta_hat = pos_mean[i].reshape(-1, 1)
            theta_sampled = new_samples[i] # num_samples by k
            avail_items = np.setdiff1d(np.arange(self.max_items), tts.item_ids[i, :item_idx])
            p_hat = norm.cdf(self.alphas[avail_items, :] @ theta_hat.flatten() + self.intercepts[avail_items]) # j-dim vec
            p_sampled = norm.cdf(self.alphas[avail_items, :] @ theta_sampled.T + self.intercepts[avail_items].reshape(-1,1)) # j * mc_samples
            kl_info = p_hat * np.mean(
                np.log(1/p_sampled * p_hat.reshape(-1, 1)), axis=1) + (1-p_hat) * np.mean(
                np.log(1/(1-p_sampled) * (1-p_hat).reshape(-1, 1)), axis=1)
            result[i] = avail_items[np.argmax(kl_info)]
        return result

    def select_kl_pos(self, item_idx: int, tts: TestTakers):
        """Select Items Based on KL Distance Between Subsequent Posteriors"""
        result = np.zeros(self.n)
        all_items = np.arange(self.max_items)
        pos_pred = tts.get_pos_pred(item_idx, self.mc_samples, self.alphas, self.intercepts)  # n by num_avail_items
        new_samples = tts.get_factor_samples(self.mc_samples)  # n by num_samples by k
        for i in range(self.n):
            pred_hat = pos_pred[i]
            theta_sampled = new_samples[i]
            avail_items = all_items[~np.in1d(all_items, tts.item_ids[i, :item_idx])]
            p_sampled = norm.cdf(self.alphas[avail_items, :] @ theta_sampled.T + self.intercepts[avail_items].reshape(-1,1)) # j * mc_samples
            kl_term1 = pred_hat * np.mean(np.log(1/p_sampled * pred_hat.reshape(-1,1)), axis=1)
            kl_term2 = (1-pred_hat) * np.mean(np.log(1/(1-p_sampled) * (1-pred_hat).reshape(-1,1)), axis=1)
            result[i] = avail_items[np.argmax(kl_term1+kl_term2)]
        return result

    def select_mi(self, item_idx: int,  tts: TestTakers):
        """Select Items Based on Mutual Information"""
        result = np.zeros(self.n)
        all_items = np.arange(self.max_items)
        pos_pred = tts.get_pos_pred(item_idx, self.alphas, self.intercepts)  # n by num_avail_items
        for i in range(self.n):
            pred_hat = pos_pred[i]
            avail_items = all_items[~np.in1d(all_items, tts.item_ids[i, :item_idx])]
            theta_sampled0 = tts.get_next_factor_samples(i, 0, self.mc_samples, self.alphas[avail_items,:],
                                                        self.intercepts[avail_items]) # (avial_items, mc_samples, k)
            theta_sampled1 = tts.get_next_factor_samples(i, 1, self.mc_samples, self.alphas[avail_items, :],
                                                         self.intercepts[avail_items])  # (avial_items, mc_samples, k)
            p_sampled0 = norm.cdf(
                np.squeeze(np.matmul(theta_sampled0, self.alphas[avail_items, :, np.newaxis]), axis=-1
                           ) +self.intercepts[avail_items].reshape(-1,1)) # j by mc samples
            p_sampled1 = norm.cdf(
                np.squeeze(np.matmul(theta_sampled1, self.alphas[avail_items, :, np.newaxis]), axis=-1
                           ) + self.intercepts[avail_items].reshape(-1, 1))  # j by mc samples
            mi_term1 = pred_hat * np.mean(np.log(p_sampled1 / pred_hat.reshape(-1, 1)), axis=1)
            mi_term2 = (1 - pred_hat) * np.mean(np.log((p_sampled0/ (1 - pred_hat).reshape(-1, 1)), axis=1))
            result[i] = avail_items[np.argmax(mi_term1 + mi_term2)]
        return result

    def select_items(self, item_idx, tts):
        """Select Next Items Depending on Given Selection Criterion for each test taker"""
        if self.sc == "kl_eap":
            return self.select_kl_eap(item_idx, tts)
        elif self.sc == "kl_pos":
            return self.select_kl_pos(item_idx, tts)
        else:
            return self.select_mi(item_idx, tts)









