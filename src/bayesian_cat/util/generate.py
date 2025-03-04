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
    alphas = np.random.multivariate_normal(np.zeros(k), np.identity(k), m)
    if zero_entries:
        assert zero_entries < k, "number of zero entries must be less than k"
        for i in range(m):
            zero_draws = np.random.choice(range(0, k), size= zero_entries, replace=False)
            alphas[i, zero_draws] = 0
    intercepts = np.random.uniform(-1.5, 1.5, m)
    true_params = {"alphas": alphas, "thetas": thetas, "intercepts": intercepts}
    return true_params


def mirt_parameters_grid(n: int, m: int, k: int, random_state: int,
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
    grid_size = 7/m  #assume alphas bounded between (-3.5, 3.5)
    alphas = np.zeros((m, k))
    for i in range(k):
        alphas[: , i] = np.random.permutation(np.arange(-3.5, 3.5, grid_size))

    if zero_entries:
        assert zero_entries < k, "number of zero entries must be less than k"
        for i in range(m):
            zero_draws = np.random.choice(range(0, k), size= zero_entries, replace=False)
            alphas[i, zero_draws] = 0
    intercepts = np.random.uniform(-1.5, 1.5, m)
    true_params = {"alphas": alphas, "thetas": thetas, "intercepts": intercepts}
    return true_params


def mirt_parameters_grid_mixed(n: int, m: int, k: int, random_state: int,
                        zero_entries: typing.List[int]) -> typing.Dict[str, typing.Any]:
    """Generate paraemters for an MIRT probit model.

    Args:
        n: num of students
        m: num of items
        k: num of dimensions
        zero_entries: number of allowable zeros per dimenion
        random_state: random state
    Returns:
        Dictionary with generated parameter values, including
            * alphas  m by k array,
            * thetas: n by k array
            * intercepts: m-dim vec
    """
    np.random.seed(random_state)
    thetas = np.random.multivariate_normal(np.zeros(k), np.identity(k), n)
    grid_size = 7/m  #assume alphas bounded between (-3.5, 3.5)
    alphas = np.zeros((m, k))
    for i in range(k):
        alphas[: , i] = np.random.permutation(np.arange(-3.5, 3.5, grid_size))
    if zero_entries:
        for i in range(m):
            zero_num = np.random.choice(zero_entries, size= 1, replace=False)
            zero_draws = np.random.choice(range(0, k), size= zero_num, replace=False)
            alphas[i, zero_draws] = 0
    intercepts = np.random.uniform(-1.5, 1.5, m)
    true_params = {"alphas": alphas, "thetas": thetas, "intercepts": intercepts}
    return true_params


def mirt_standarized_parameters_grid_mixed(n: int, m: int, k: int, random_state: int,
                        zero_entries: typing.List[int]) -> typing.Dict[str, typing.Any]:
    """Generate standarized paraemters for an MIRT probit model.

    Args:
        n: num of students
        m: num of items
        k: num of dimensions
        zero_entries: number of allowable zeros per dimenion
        random_state: random state
    Returns:
        Dictionary with generated parameter values, including
            * alphas  m by k array,
            * thetas: n by k array
            * intercepts: m-dim vec
    """
    np.random.seed(random_state)
    thetas = np.random.multivariate_normal(np.zeros(k), np.identity(k), n)
    alphas = np.zeros((m, k))
    for i in range(k):
        neg_num = m//2
        pos_num = m - neg_num
        alphas[: , i] = np.random.permutation(np.concatenate([np.linspace(-3, -0.2, neg_num),np.linspace(0.2, 3, pos_num)]))
    if zero_entries:
        for i in range(m):
            zero_num = np.random.choice(zero_entries, size= 1, replace=False)
            zero_draws = np.random.choice(range(0, k), size= zero_num, replace=False)
            alphas[i, zero_draws] = 0
    intercepts = np.random.uniform(-1.5, 1.5, m)
    true_params = {"alphas": alphas, "thetas": thetas, "intercepts": intercepts}
    return true_params
