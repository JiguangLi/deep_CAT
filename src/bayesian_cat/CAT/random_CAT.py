import typing
import numpy as np
from tqdm import tqdm
import statsmodels.api as sm
from scipy.stats import norm
import warnings

class RandomCAT:

    def __init__(
            self,
            true_thetas: np.ndarray,
            test_bank: typing.Dict[str, np.ndarray],
            max_items: int,
            num_simulations: int,
            starting_item: typing.Optional[int] = None,
            random_state: int = 42,
    ):
        """
        Create a Randomized Computerized Testing Object for Simulation, served as a baseline (frequntist approach)

            Args:
                true_thetas: Shape: (n, k), latent factors of new test takers
                test_bank: dictionary with loading matrix "alphas", and "intercepts" for the items in the test bank
                max_items: maximum numbers to test, currently no stopping criterions for the simulation
                num_simulations: how many rounds of simulations
                starting_item: first item to start or we do self select
                random_state: seed
            Raises:
                ValueError: if given inputs do not meet object expectations.
        """
        self.seed = random_state
        self.rng = np.random.default_rng(random_state)
        self.true_thetas = true_thetas
        self.n = true_thetas.shape[0]
        self.k = true_thetas.shape[1]
        self.alphas = test_bank["alphas"]
        self.intercepts = test_bank["intercepts"]
        self.max_items = max_items
        self.num_sims = num_simulations
        self.m = self.alphas.shape[0]
        self.theta_updates = np.zeros((self.max_items, self.n, self.k))
        self.mse_array = np.zeros((self.num_sims, self.k))
        if starting_item:
            self.start = starting_item
        else:
            self.start = self.pick_starting_item()
        item_array = np.arange(500)
        self.avail_items = item_array[item_array!=self.start]

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
        for i in tqdm(range(self.num_sims)):
            items = self.random_select()
            theta_est = np.zeros((self.n, self.k))
            excluded_tts = []
            for j in range(self.n):
                prob_vec = norm.cdf(np.matmul(self.alphas[items[j]], self.true_thetas[j]) + self.intercepts[items[j]])
                response = (self.rng.uniform(0, 1, self.max_items) < prob_vec).astype(int)
                try: # in case model not identified
                    with warnings.catch_warnings():
                        warnings.filterwarnings('error')  # Treat warnings as errors
                        probit_model = sm.Probit(response, self.alphas[items[j]], offset=self.intercepts[items[j]])
                        probit_result = probit_model.fit(disp=0)
                        theta_est[j] = probit_result.params
                except Exception as e:
                    excluded_tts.append(j)
            mask = np.ones(self.n, dtype=bool)
            mask[excluded_tts] = False
            self.mse_array[i] = np.mean((self.true_thetas[mask]-theta_est[mask])**2, axis=0)

    def random_select(self):
        """"Random Selection"""
        result = np.zeros((self.n, self.max_items), dtype=int)
        result[:, 0] = np.ones(self.n, dtype=int)*self.start
        draws = np.array([self.rng.choice(self.avail_items, self.max_items-1, replace=False) for _ in range(self.n)])
        result[:, 1:] = draws
        return result
















