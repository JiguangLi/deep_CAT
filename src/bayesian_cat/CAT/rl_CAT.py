import typing
import numpy as np
from .test_taker import TestTakers
from tqdm import tqdm
from scipy.stats import norm
import multiprocessing


class RLCAT:

    def __init__(
            self,
            true_thetas: np.ndarray,
            test_bank: typing.Dict[str, np.ndarray],
            max_items: int,
            mc_samples: int,
            sir_large_sample: int,
            starting_item: typing.Optional[int] = None,
            rollout_h: int = None,
            num_workers: int = 8,
            random_state: int = 42,
    ):
        """
        Create a Bayesian Computerized Testing Object for Simulation

            Args:
                true_thetas: Shape: (n, k), latent factors of new test takers
                test_bank: dictionary with loading matrix "alphas", and "intercepts" for the items in the test bank
                max_items: maximum numbers to test, currently no stopping criterions for the simulation
                mc_samples: number of mc samples
                sir_large_sample: number of large samples for posterior reweighting
                starting_item: first item to start or we do self select
                rollout_h: rollout horizon, if none it is maximim items
                num_workers: number of cores for multiprocessing
                random_state: seed
            Raises:
                ValueError: if given inputs do not meet object expectations.
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
        self.m = self.alphas.shape[0]
        self.theta_updates = np.zeros((self.max_items, self.n, self.k))
        self.item_selections = np.zeros((self.n, self.max_items))
        if starting_item:
            self.start = starting_item
        else:
            self.start = self.pick_starting_item()
        if rollout_h:
            self.rollout_h = rollout_h
        else:
            self.rollout_h = max_items
        self.sir_large_samples = sir_large_sample
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

    # def pos_reweighting_by_action(self, pos_samples: np.ndarray, action:int):
    #     """SIR by action"""
    #     alpha, intercept = self.alphas[action], self.intercepts[action]
    #     prob1 = norm.cdf(pos_samples@alpha + intercept)
    #     prob0 = 1 - prob1
    #     weight1 = prob1 / (np.sum(prob1))  # length sir_large_samples
    #     weight0 = prob0 / (np.sum(prob0))
    #     indices0 = np.random.choice(a=self.sir_large_samples, size=self.mc_samples, p=weight0)
    #     indices1 = np.random.choice(a=self.sir_large_samples, size=self.mc_samples, p=weight1)
    #     theta0_sampled = pos_samples[indices0]
    #     theta1_sampled = pos_samples[indices1]
    #     pred_hat = np.mean(prob1)
    #     p_sampled1 = norm.cdf(theta1_sampled @ alpha + intercept)
    #     p_sampled0 = 1 - norm.cdf(theta0_sampled @ alpha + intercept)
    #     mi_term1 = pred_hat * np.mean(np.log(p_sampled1 / pred_hat))
    #     mi_term2 = (1 - pred_hat) * np.mean(np.log(p_sampled0 / (1 - pred_hat)))
    #     reward = mi_term1 + mi_term2
    #     # get new posteriors
    #     new_pos = np.zeros((self.mc_samples, self.k))
    #     n1_samples = int(self.mc_samples*pred_hat)
    #     n0_samples = self.mc_samples - n1_samples
    #     new_pos[: n1_samples] = theta1_sampled[np.random.choice(range(0, self.mc_samples), size=n1_samples,
    #                                                             replace=False)]
    #     new_pos[-n0_samples:] = theta0_sampled[np.random.choice(range(0, self.mc_samples), size=n0_samples,
    #                                                             replace=False)]
    #     return reward, new_pos

    # def pos_reweighting_all(self, pos_samples: np.ndarray, avail_actions: np.array):
    #     """SIR for all actions"""
    #     alphas, intercepts = self.alphas[avail_actions], self.intercepts[avail_actions]
    #     prob1 = norm.cdf(alphas @ pos_samples.T + intercepts.reshape(-1, 1))  # j *mc_samples
    #     prob0 = 1 - prob1
    #     weight1 = prob1 / (np.sum(prob1, axis=1).reshape(-1, 1))  # j * sir_large_samples
    #     weight0 = prob0 / (np.sum(prob0, axis=1).reshape(-1, 1))
    #     m = len(avail_actions)
    #     theta1_sampled, theta0_sampled = np.zeros((m, self.mc_samples, self.k)), np.zeros((m, self.mc_samples, self.k))
    #     for j in range(m):
    #         indices0 = np.random.choice(a=self.mc_samples, size=self.mc_samples, p=weight0[j])
    #         indices1 = np.random.choice(a=self.mc_samples, size=self.mc_samples, p=weight1[j])
    #         theta0_sampled[j] = pos_samples[indices0]
    #         theta1_sampled[j] = pos_samples[indices1]
    #     p_sampled0 = 1 - norm.cdf(
    #         np.squeeze(np.matmul(theta0_sampled, alphas[:, :, np.newaxis]), axis=-1
    #                    ) + intercepts.reshape(-1, 1))  # j by mc samples
    #     p_sampled1 = norm.cdf(
    #         np.squeeze(np.matmul(theta1_sampled,  alphas[:, :, np.newaxis]), axis=-1
    #                    ) + intercepts.reshape(-1, 1))  # j by mc samples
    #     pred_hat = np.mean(prob1, axis=1)
    #     mi_term1 = pred_hat * np.mean(np.log(p_sampled1 / pred_hat.reshape(-1, 1)), axis=1)
    #     mi_term2 = (1 - pred_hat) * np.mean(np.log(p_sampled0 / (1 - pred_hat).reshape(-1, 1)), axis=1)
    #     rewards = mi_term1 + mi_term2
    #     max_reward = np.max(rewards)
    #     arg_max_reward = np.argmax(rewards)
    #     best_action = avail_actions[np.argmax(rewards)]
    #     # get new posteriors
    #     new_pos = np.zeros((self.mc_samples, self.k))
    #     n1_samples = int(self.mc_samples * pred_hat[arg_max_reward])
    #     n0_samples = self.mc_samples - n1_samples
    #     new_pos[: n1_samples] = theta1_sampled[arg_max_reward][np.random.choice(range(0, self.mc_samples),
    #                                                                             size=n1_samples,
    #                                                                             replace=False)]
    #     new_pos[-n0_samples:] = theta0_sampled[arg_max_reward][np.random.choice(range(0, self.mc_samples),
    #                                                                             size=n0_samples,
    #                                                                             replace=False)]
    #     return best_action, max_reward, new_pos

    # def rollout(self, pos_samples: np.ndarray, action: int, avail_items, item_index: int):
    #     """Approximate Value function using Rollout"""
    #     reward, pos_samples = self.pos_reweighting_by_action(pos_samples, action)
    #     num_steps = min(self.rollout_h, self.max_items - item_index - 2)
    #     for s in range(num_steps):
    #         avail_items = avail_items[avail_items != action]
    #         action, new_reward, pos_samples = self.pos_reweighting_all(pos_samples, avail_items)
    #         new_reward  = max(new_reward, 0)
    #         reward += new_reward
    #        # print("action {}, rollout step {}, reward {}".format(action, s, reward))
    #     return reward

    # def select_items(self, item_idx):
    #     """Select items for each person"""
    #     result = np.zeros(self.n, dtype=int)
    #     pos_samples = self.tts.get_factor_samples(self.sir_large_samples) # (n, large_samples, k)
    #     for i in tqdm(range(self.n)):
    #         avail_items = np.setdiff1d(np.arange(self.m), self.tts.item_ids[i, :item_idx])
    #         arguments = [(pos_samples[i], j, avail_items, item_idx) for j in avail_items]
    #         with multiprocessing.Pool(self.num_workers) as pool:
    #             out = pool.starmap(self.rollout, arguments)
    #         #print("overall horizon"+str(item_idx))
    #         #print(i)
    #         #print(out)
    #         result[i] = avail_items[out.index(max(out))]
    #     return result

    def pos_reweighting_by_action(self, pos_samples: np.ndarray, action: int):
        """SIR by action"""
        alpha, intercept = self.alphas[action], self.intercepts[action]
        prob1 = norm.cdf(pos_samples @ alpha + intercept)
        prob0 = 1 - prob1
        weight1 = prob1 / (np.sum(prob1))  # length sir_large_samples
        weight0 = prob0 / (np.sum(prob0))
        indices0 = np.random.choice(a=self.sir_large_samples, size=self.mc_samples, p=weight0)
        indices1 = np.random.choice(a=self.sir_large_samples, size=self.mc_samples, p=weight1)
        theta0_sampled = pos_samples[indices0]
        theta1_sampled = pos_samples[indices1]
        pred_hat = np.mean(prob1)
        p_sampled1 = norm.cdf(theta1_sampled @ alpha + intercept)
        p_sampled0 = 1 - norm.cdf(theta0_sampled @ alpha + intercept)
        mi_term1 = pred_hat * np.mean(np.log(p_sampled1 / pred_hat))
        mi_term2 = (1 - pred_hat) * np.mean(np.log(p_sampled0 / (1 - pred_hat)))
        reward = mi_term1 + mi_term2
        # get new posteriors
        new_pos = np.zeros((self.mc_samples, self.k))
        n1_samples = int(self.mc_samples * pred_hat)
        n0_samples = self.mc_samples - n1_samples
        new_pos[: n1_samples] = theta1_sampled[np.random.choice(range(0, self.mc_samples), size=n1_samples,
                                                                replace=False)]
        new_pos[-n0_samples:] = theta0_sampled[np.random.choice(range(0, self.mc_samples), size=n0_samples,
                                                                replace=False)]
        return reward, new_pos

    def pos_reweighting_all(self, og_samples: np.ndarray, avail_actions: np.array,
                            prob0: np.ndarray, prob1:np.ndarray, pred_hat: np.array):
        """SIR for all actions

        Use current posterior to
        Sample new posterior using the OG posterior
        """
        alphas, intercepts = self.alphas[avail_actions], self.intercepts[avail_actions]
        new_weight1 = prob1[avail_actions, :] # j * sir_large_samples
        new_weight0 = prob0[avail_actions, :]
        new_weight1 = new_weight1/(np.sum(new_weight1, axis=1).reshape(-1, 1))
        new_weight0 = new_weight0/(np.sum(new_weight0, axis=1).reshape(-1, 1))
        m = len(avail_actions)
        theta1_sampled, theta0_sampled = np.zeros((m, self.mc_samples, self.k)), np.zeros((m, self.mc_samples, self.k))
        for j in range(m):
            indices0 = np.random.choice(a=self.sir_large_samples, size=self.mc_samples, p=new_weight0[j])
            indices1 = np.random.choice(a=self.sir_large_samples, size=self.mc_samples, p=new_weight1[j])
            theta0_sampled[j] = og_samples[indices0]
            theta1_sampled[j] = og_samples[indices1]
        p_sampled0 = 1 - norm.cdf(
            np.squeeze(np.matmul(theta0_sampled, alphas[:, :, np.newaxis]), axis=-1
                       ) + intercepts.reshape(-1, 1))  # j by mc samples
        p_sampled1 = norm.cdf(
            np.squeeze(np.matmul(theta1_sampled,  alphas[:, :, np.newaxis]), axis=-1
                       ) + intercepts.reshape(-1, 1))  # j by mc samples
        mi_term1 = pred_hat * np.mean(np.log(p_sampled1 / pred_hat.reshape(-1, 1)), axis=1)
        mi_term2 = (1 - pred_hat) * np.mean(np.log(p_sampled0 / (1 - pred_hat).reshape(-1, 1)), axis=1)
        rewards = mi_term1 + mi_term2
        max_reward = np.max(rewards)
        arg_max_reward = np.argmax(rewards)
        best_action = avail_actions[np.argmax(rewards)]
        # get new posteriors
        new_pos = np.zeros((self.mc_samples, self.k))
        n1_samples = int(self.mc_samples * pred_hat[arg_max_reward])
        n0_samples = self.mc_samples - n1_samples
        new_pos[: n1_samples] = theta1_sampled[arg_max_reward][np.random.choice(range(0, self.mc_samples),
                                                                                size=n1_samples,
                                                                                replace=False)]
        new_pos[-n0_samples:] = theta0_sampled[arg_max_reward][np.random.choice(range(0, self.mc_samples),
                                                                                size=n0_samples,
                                                                                replace=False)]
        return best_action, max_reward, new_pos

    def rollout(self, pos_samples: np.ndarray, action: int, avail_items, item_index: int,
                prob0: np.ndarray, prob1: np.ndarray):
        """Approximate Value function using Rollout

        Use Current Sample To Make Predictions of Future Mixture of Weight
        Sample Conditional (on Y) posterior from the OG posterior
        """
        reward, cur_pos_samples = self.pos_reweighting_by_action(pos_samples, action)
        num_steps = min(self.rollout_h, self.max_items - item_index - 2)
        for s in range(num_steps):
            avail_items = avail_items[avail_items != action]
            pred_hat = np.mean(
                norm.cdf(self.alphas[avail_items] @ cur_pos_samples.T + self.intercepts[avail_items].reshape(-1, 1)),
                axis=1)
            action, new_reward, cur_pos_samples = self.pos_reweighting_all(
                pos_samples, avail_items, prob0, prob1, pred_hat)
            new_reward = max(new_reward, 0)
            reward += new_reward
        # print("action {}, rollout step {}, reward {}".format(action, s, reward))
        return reward

    def select_items(self, item_idx):
        """Select items for each person"""
        result = np.zeros(self.n, dtype=int)
        pos_samples = self.tts.get_factor_samples(self.sir_large_samples)  # (n, large_samples, k)
        for i in tqdm(range(self.n)):
            avail_items = np.setdiff1d(np.arange(self.m), self.tts.item_ids[i, :item_idx])
            prob1 = norm.cdf(self.alphas@pos_samples[i].T + self.intercepts.reshape(-1, 1))  # j * sir_large_samples
            prob0 = 1 - prob1
            arguments = [(pos_samples[i], j, avail_items, item_idx, prob0, prob1) for j in avail_items]
            with multiprocessing.Pool(self.num_workers) as pool:
                out = pool.starmap(self.rollout, arguments)
            # print("overall horizon"+str(item_idx))
            # print(i)
            # print(out)
            result[i] = avail_items[out.index(max(out))]
        return result

    def simulate(self):
        """For each test taker, simulate their responses to selected items and record theta_updates"""
        start_items = np.array([self.start]*self.n)
        self.tts.answer_items(self.alphas[start_items], self.intercepts[start_items], start_items)
        self.tts.get_pos_mean(0, self.mc_samples)
        for j in range(1, self.max_items):
            print("horizon", j+1)
            new_items = self.select_items(j)
            self.tts.answer_items(self.alphas[new_items], self.intercepts[new_items], new_items)
            self.tts.get_pos_mean(j, self.mc_samples)
        self.theta_updates = self.tts.pos_mean
        self.item_selections = self.tts.item_ids











