import pathlib
import yaml
import bayesian_cat as bcat
import argparse
import typing
import pickle
import numpy as np

def parse_args(verbose: bool = False) -> typing.Dict[str, typing.Any]:
    """Get command line arguments, with defaults set for local testing."""
    # parse command line arguments first
    parser = argparse.ArgumentParser("Simulate MIRT parameters in a test bank")
    parser.add_argument("--config_filepath",
                        default= pathlib.Path.home().joinpath(pathlib.Path("bayesian-cat", "config", "config.yaml")))
    parser.add_argument("--verbose", default=True)
    parser.add_argument("--n", default=500)
    parser.add_argument("--m", default=150)
    parser.add_argument("--k", default=5)
    parser.add_argument("--zero_entries", default=[1,2,3,4], help="factor loadings allowed to be zero")
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

def simulate_model_params(n, m, k):
    np.random.seed(0)
    thetas = np.random.multivariate_normal(np.zeros(k), np.identity(k), n)
    alphas = np.zeros((m, k), dtype=float)
    # main factor is draw from uniform
    alphas[:, 0] = np.random.uniform(-3, 3, size=m)
    # fill secondary and third factors
    rows = np.arange(m)
    cols12 = np.random.randint(1, 3, size=m)      # 1 or 2
    alphas[rows, cols12] = np.random.uniform(-3, 3, size=m)
    # fill fourth and fith factor  
    cols34 = np.random.randint(3, 5, size=m)      # 3 or 4
    alphas[rows, cols34] =  np.random.uniform(-3, 3, size=m)
    # lower triangularize the array to ensure identification
    alphas = np.tril(alphas)           # keep lower triangle (incl. diagonal)
    np.fill_diagonal(alphas, 1.0)  # postive diagonal
    intercepts = np.random.uniform(-1.5, 1.5, m)
    true_params = {"alphas": alphas, "thetas": thetas, "intercepts": intercepts}
    return true_params

if __name__ == "__main__":
    # data setup
    arguments = parse_args()
    n, m, k, zero_entries = arguments["n"], arguments["m"], arguments["k"], arguments["zero_entries"]
    #sim_params = bcat.util.mirt_standarized_parameters_grid_mixed(n, m, k, arguments["seed"], zero_entries)
    sim_params = simulate_model_params(n, m, k)
    max_zero = len(zero_entries)
    filename = "x01_lower_triag_{}_{}_{}_{}.pickle".format(n, m, k, max_zero)
    data_output_dir = pathlib.Path(arguments["sim_data_dir"])
    with open(data_output_dir.joinpath(filename), 'wb') as handle:
        pickle.dump(sim_params, handle, pickle.HIGHEST_PROTOCOL)

