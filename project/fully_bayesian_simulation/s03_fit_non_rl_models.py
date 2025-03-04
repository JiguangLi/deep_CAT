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
    parser.add_argument("--input_test_name", default="x10_s01_mixed_grid_500_150_3_2.pickle")
    parser.add_argument("--input_params_name", default="mcmc_x10_s01_mixed_grid_500_150_3_2.pickle")
    parser.add_argument("--max_items", default=30, help="after how many items to stop")
    parser.add_argument("--sir_start", default=1, help="after how many items to do posterior reweighting")
    parser.add_argument("--mc_samples", default=1)
    parser.add_argument("--loading_pos_sample", type=int, default=500)
    parser.add_argument("--sir_large_samples", default=750)
    parser.add_argument("--sir_samples", default=1)
    parser.add_argument("--dim", default=3)
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
    # parse arguments
    arguments = parse_args()
    # read test bank & mcmc model
    data_input_dir = pathlib.Path(arguments["sim_data_dir"], arguments["input_test_name"])
    with open(data_input_dir, 'rb') as handle:
        test_bank = pickle.load(handle)
    mcmc_input_dir = pathlib.Path(arguments["mcmc_models_dir"], arguments["input_params_name"])
    with open(mcmc_input_dir, 'rb') as handle:
        mcmc_params = pickle.load(handle)
    pos_alphas = mcmc_params.params["alphas"][-arguments["loading_pos_sample"]:]
    pos_alphas = - pos_alphas / 1.6
    pos_intercepts = mcmc_params.params["intercepts"][-arguments["loading_pos_sample"]:, :, 0]
    pos_intercepts = pos_intercepts / 1.6
    sir_params = {"sir_start": arguments["sir_start"], "large_samples": arguments["sir_large_samples"],
                  "samples": arguments["sir_samples"]}
    # fit models
    selection_rules = ["kl_eap","kl_pos", "mi_sir", "predictive_variance_e"]
    # selection_rules = ["kl_eap", "kl_pos", "mi_sir", "predictive_variance_e"]
    for rule in selection_rules:
        print(rule)
        model = bcat.FullyBayesianCAT(true_thetas= test_bank["thetas"][:1],
                                      test_bank= {"alphas": pos_alphas, "intercepts": pos_intercepts},
                                      selection_criterion=rule,
                                      max_items=arguments["max_items"],
                                      mc_samples=arguments["mc_samples"],
                                      plugin_response= True,
                                      true_response= test_bank["response"].values[:1],
                                      starting_item=None,
                                      sir_params=sir_params,
                                      num_workers=arguments["num_workers"],
                                      random_state=arguments["seed"])
        model.simulate()
        #filename = "Bayesian_x10_s03_3factor_mixed_s03_{}_{}.pickle".format(rule, arguments["max_items"])
        filename = "Bayesian_x10_s03_speed_testing_{}_{}.pickle".format(rule, arguments["max_items"])
        model_output_dir = pathlib.Path(arguments["sim_models_dir"])
        with open(model_output_dir.joinpath(filename), 'wb') as handle:
            pickle.dump(model, handle, pickle.HIGHEST_PROTOCOL)
        del model
        # partially Bayesian
        # model = bcat.BayesianCAT(true_thetas=test_bank["thetas"],
        #                          test_bank= {"alphas": np.mean(pos_alphas, axis=0),
        #                                      "intercepts": np.mean(pos_intercepts, axis=0)},
        #                          selection_criterion=rule,
        #                          max_items=arguments["max_items"],
        #                          mc_samples=arguments["mc_samples"],
        #                          plugin_response=True,
        #                          true_response=test_bank["response"].values,
        #                          starting_item=None,
        #                          sir_params=sir_params,
        #                          num_workers=arguments["num_workers"],
        #                          random_state=arguments["seed"])
        # model.simulate()
        # filename = "Partially_Bayesian_x10_s03_3factor_mixed_s03_{}_{}.pickle".format(rule, arguments["max_items"])
        # model_output_dir = pathlib.Path(arguments["sim_models_dir"])
        # with open(model_output_dir.joinpath(filename), 'wb') as handle:
        #     pickle.dump(model, handle, pickle.HIGHEST_PROTOCOL)
        # del model

