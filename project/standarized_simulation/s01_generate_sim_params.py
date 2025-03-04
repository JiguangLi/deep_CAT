import pathlib
import yaml
import bayesian_cat as bcat
import argparse
import typing
import pickle


def parse_args(verbose: bool = False) -> typing.Dict[str, typing.Any]:
    """Get command line arguments, with defaults set for local testing."""
    # parse command line arguments first
    parser = argparse.ArgumentParser("Simulate MIRT parameters in a test bank")
    parser.add_argument("--config_filepath",
                        default= pathlib.Path.home().joinpath(pathlib.Path("bayesian-cat", "config", "config.yaml")))
    parser.add_argument("--verbose", default=True)
    parser.add_argument("--n", default=500)
    parser.add_argument("--m", default=200)
    parser.add_argument("--k", default=5)
    parser.add_argument("--zero_entries", default=[3,4])
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


if __name__ == "__main__":
    # data setup
    arguments = parse_args()
    n, m, k, zero_entries = arguments["n"], arguments["m"], arguments["k"], arguments["zero_entries"]
    sim_params = bcat.util.mirt_standarized_parameters_grid_mixed(n, m, k, arguments["seed"], zero_entries)
    max_zero = max(zero_entries)
    filename = "x09_s01_mixed_grid_{}_{}_{}_{}.pickle".format(n, m, k, max_zero)
    data_output_dir = pathlib.Path(arguments["sim_data_dir"])
    with open(data_output_dir.joinpath(filename), 'wb') as handle:
        pickle.dump(sim_params, handle, pickle.HIGHEST_PROTOCOL)

