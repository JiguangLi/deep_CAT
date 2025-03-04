import pathlib
import yaml
import bayesian_cat as bcat
import argparse
import typing
import pickle
import pandas as pd
import numpy as np


def parse_args(verbose: bool = False) -> typing.Dict[str, typing.Any]:
    """Get command line arguments, with defaults set for local testing."""
    # parse command line arguments first
    parser = argparse.ArgumentParser("Deep Q Network for Item selection")
    parser.add_argument("--config_filepath",
                        default= pathlib.Path.home().joinpath(pathlib.Path("bayesian-cat", "config", "config.yaml")))
    parser.add_argument("--verbose", default=True)
    parser.add_argument("--input_params_name", default="px_em_item_params.feather")
    parser.add_argument("--input_response_name", default="grade8_item_responses.feather")
    parser.add_argument("--max_items", default=50, help="after how many items to stop")
    parser.add_argument("--mc_samples", default=500)
    parser.add_argument("--network_version", default="v4")
    parser.add_argument("--pred_type", default="detailed")
    parser.add_argument("--num_workers", default=8)
    parser.add_argument("--dim", default=4)
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
    # Input Testbank
    arguments = parse_args()
    loadings_dir = pathlib.Path(arguments["dese_models_dir"], arguments["input_params_name"])
    response_dir = pathlib.Path(arguments["dese_models_dir"], arguments["input_response_name"])
    loading_df = pd.read_feather(loadings_dir)
    response_df = pd.read_feather(response_dir)
    test_params = {"alphas": loading_df.values[:, :-1], "intercepts": loading_df.values[:, -1],
                   "thetas": np.zeros((response_df.shape[0], arguments["dim"]))}
    # Input trained NN directory
    file_names = [
        "online-Q-network-v4-4-factor/NN_px_em_item_params_150000_first0-1_detailed_v4_1.5e-05_128_30000_800000_0.15/fn_28000.pt",
        ]

    for filename in file_names:
        nn_input_dir = pathlib.Path(arguments["dese_models_dir"]).joinpath(filename)
        # similuate
        model = bcat.OnlineDeepQCAT(
            true_thetas=test_params["thetas"],
            test_bank=test_params,
            nn_model_dir = nn_input_dir,
            max_items=arguments["max_items"],
            mc_samples= arguments["mc_samples"],
            network_version= arguments["network_version"],
            pred_type= arguments["pred_type"],
            plugin_response= True,
            true_response= response_df.values,
            num_workers=arguments["num_workers"],
            random_state=arguments["seed"]
        )
        model.simulate()
        filename = "dese_s06_QCAT_{}_{}_{}_{}_final.pickle".format(arguments["max_items"],filename.split("/")[0], filename.split("/")[1], filename.split("/")[2] )
        model_output_dir = pathlib.Path(arguments["dese_models_dir"]).joinpath("Q-CAT")
        with open(model_output_dir.joinpath(filename), 'wb') as handle:
            pickle.dump(model, handle, pickle.HIGHEST_PROTOCOL)
        del model


