import typing
import numpy as np
import pandas as pd
from scipy.stats import norm


def probit_mirt_binary_data(params: typing.Dict[str, typing.Any], seed=42):
    """Generate Probit MIRT Binary Data"""
    np.random.seed(seed)
    linear_term = np.matmul(params["thetas"], params["alphas"].T) + params["intercepts"]
    prob_matrix = norm.cdf(linear_term)
    n, m = prob_matrix.shape
    result_matrix = (np.random.uniform(0, 1, (n, m)) < prob_matrix).astype(int)
    col_names = ["item" + str(i) for i in range(1, m + 1)]
    df = pd.DataFrame(data=result_matrix, columns=col_names)
    return df