import pathlib
import pandas as pd
import yaml
import bayesian_cat as bcat
import argparse
import typing
import pickle


def parse_args(verbose: bool = False) -> typing.Dict[str, typing.Any]:
    """Get command line arguments, with defaults set for local testing."""
    # parse command line arguments first
    parser = argparse.ArgumentParser("Standard CAT Algo Without Subsetting")
    parser.add_argument("--config_filepath",
                        default= pathlib.Path.home().joinpath(pathlib.Path("bayesian-cat", "config", "config.yaml")))
    parser.add_argument("--verbose", default=True)
    parser.add_argument("--input_data_name", default="x01_lower_triag_500_150_5_4.pickle")
    parser.add_argument("--max_items", default=50, help="after how many items to stop")
    parser.add_argument("--subset", default=[0,1,2], help="subset factors of interest")
    parser.add_argument("--sir_start", default=1, help="after how many items to do posterior reweighting")
    parser.add_argument("--mc_samples", default=500)
    parser.add_argument("--sir_large_samples", default=500)
    parser.add_argument("--sir_samples", default=500)
    parser.add_argument("--num_workers", default=8)
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
    sir_params = {"sir_start": arguments["sir_start"], "large_samples": arguments["sir_large_samples"],
                  "samples": arguments["sir_samples"]}
    # fit models
    selection_rules = ["kl_eap", "kl_pos", "mi_sir", "predictive_variance_e"]
    for rule in selection_rules:
        print(rule)
        model = bcat.BayesianCAT(true_thetas=test_params["thetas"],
                                 test_bank=test_params,
                                 selection_criterion=rule,
                                 max_items=arguments["max_items"],
                                 mc_samples=arguments["mc_samples"],
                                 starting_item=None,
                                 sir_params=sir_params,
                                 subset= arguments["subset"],
                                 num_workers=arguments["num_workers"],
                                 random_state=arguments["seed"])
        model.simulate()
        filename = "testings_standarized_5_factor_trig_s05_{}_{}_subset.pickle".format(rule, arguments["max_items"])
        model_output_dir = pathlib.Path(arguments["sim_models_dir"])
        with open(model_output_dir.joinpath(filename), 'wb') as handle:
            pickle.dump(model, handle, pickle.HIGHEST_PROTOCOL)
        del model

