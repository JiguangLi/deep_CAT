import pathlib
import yaml
import bayesian_cat as bcat
import argparse
import typing
import pickle


def parse_args(verbose: bool = False) -> typing.Dict[str, typing.Any]:
    """Get command line arguments, with defaults set for local testing."""
    # parse command line arguments first
    parser = argparse.ArgumentParser("Fit Gibbs Samplers on QOL dataset")
    parser.add_argument("--config_filepath",
                        default= pathlib.Path.home().joinpath(pathlib.Path("bayesian-cat", "config", "config.yaml")))
    parser.add_argument("--verbose", default=True)
    parser.add_argument("--input_data_name", default="x10_s01_mixed_grid_500_150_3_2.pickle")
    parser.add_argument("--num_samples", default=5000)
    parser.add_argument("--dim", default=3)
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
    data_input_dir = pathlib.Path(arguments["sim_data_dir"], arguments["input_data_name"])
    with open(data_input_dir, 'rb') as handle:
        test_params = pickle.load(handle)
    alphas = test_params["alphas"]
    # get_loading_constraints
    loading_constraints = {}
    for i in range(alphas.shape[0]):
        for j in range(alphas.shape[1]):
            if alphas[i, j] == 0:
                loading_constraints[(i, j)] = 0
    # Get factor loading constraints
    mcmc_model = bcat.BayesianMIRT.MCMC_MIRT(
        test_params["response"],
        dim= arguments["dim"],
        num_samples= arguments["num_samples"],
        alpha_prior="normal",
        loading_constraints= loading_constraints
    )
    mcmc_model.fit()
    #save models
    model_output_dir = pathlib.Path(arguments["mcmc_models_dir"])
    model_filename = "mcmc_" + arguments["input_data_name"]
    with open(model_output_dir.joinpath(model_filename), 'wb') as handle:
        pickle.dump(mcmc_model, handle, pickle.HIGHEST_PROTOCOL)

