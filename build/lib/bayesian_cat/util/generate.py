import typing
import numpy as np


def mirt_parameters(n: int, m: int, k: int, random_state: int,
                    zero_entries: typing.Optional[int] = None) -> typing.Dict[str, typing.Any]:
    """Generate paraemters for an MIRT probit model.

    Args:
        n: num of students
        m: num of items
        k: num of dimensions
        zero_entries: number of dimensions equal to zero per dimensions
        random_state: random state
    Returns:
        Dictionary with generated parameter values, including
            * alphas  m by k array,
            * thetas: n by k array
            * intercepts: m-dim vec
    """
    np.random.seed(random_state)
    thetas = np.random.multivariate_normal(np.zeros(k), np.identity(k), n)
    alphas = np.random.multivariate_normal(np.ones(k), np.identity(k), m)
    if zero_entries:
        assert zero_entries < k, "number of zero entries must be less than k"
        for i in range(m):
            zero_draws = np.random.choice(range(0, k), size= zero_entries, replace=False)
            alphas[i, zero_draws] = 0
    intercepts = np.random.uniform(-3, 3, m)
    true_params = {"alphas": alphas.T, "thetas": thetas, "intercepts": intercepts}
    return true_params
