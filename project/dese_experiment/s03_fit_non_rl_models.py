import pathlib
import pandas as pd
import yaml
import bayesian_cat as bcat
import argparse
import typing
import pickle
import numpy as np


def parse_args(verbose: bool = False) -> typing.Dict[str, typing.Any]:
    """Get command line arguments, with defaults set for local testing."""
    # parse command line arguments first
    parser = argparse.ArgumentParser("Fit Gibbs Samplers on QOL dataset")
    parser.add_argument("--config_filepath",
                        default= pathlib.Path.home().joinpath(pathlib.Path("bayesian-cat", "config", "config.yaml")))
    parser.add_argument("--verbose", default=True)
    parser.add_argument("--input_params_name", default= "px_em_item_params.feather")
    parser.add_argument("--input_response_name", default="grade8_item_responses.feather")
    parser.add_argument("--max_items", default=51, help="after how many items to stop")
    parser.add_argument("--sir_start", default=1, help="after how many items to do posterior reweighting")
    parser.add_argument("--mc_samples", default=500)
    parser.add_argument("--sir_large_samples", default=750)
    parser.add_argument("--sir_samples", default=500)
    parser.add_argument("--dim", default=4)
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
    loadings_dir = pathlib.Path(arguments["dese_models_dir"], arguments["input_params_name"])
    response_dir = pathlib.Path(arguments["dese_models_dir"], arguments["input_response_name"])
    loading_df = pd.read_feather(loadings_dir)
    response_df = pd.read_feather(response_dir)
    test_params = {"alphas": loading_df.values[:, :-1], "intercepts": loading_df.values[:, -1],
                   "thetas": np.zeros((response_df.shape[0], arguments["dim"]))}
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
                                 plugin_response= True,
                                 true_response = response_df.values,
                                 starting_item=None,
                                 sir_params=sir_params,
                                 num_workers=arguments["num_workers"],
                                 random_state=arguments["seed"])
        model.simulate()
        filename = "dese_s03_{}_{}.pickle".format(rule, arguments["max_items"])
        model_output_dir = pathlib.Path(arguments["dese_models_dir"])
        with open(model_output_dir.joinpath(filename), 'wb') as handle:
            pickle.dump(model, handle, pickle.HIGHEST_PROTOCOL)
        del model

