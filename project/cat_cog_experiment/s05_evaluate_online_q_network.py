import pathlib
import yaml
import bayesian_cat as bcat
import argparse
import typing
import pickle
import pandas as pd


def parse_args(verbose: bool = False) -> typing.Dict[str, typing.Any]:
    """Get command line arguments, with defaults set for local testing."""
    # parse command line arguments first
    parser = argparse.ArgumentParser("Deep Q Network for Item selection")
    parser.add_argument("--config_filepath",
                        default= pathlib.Path.home().joinpath(pathlib.Path("bayesian-cat", "config", "config.yaml")))
    parser.add_argument("--verbose", default=True)
    parser.add_argument("--input_params_name", default="bifactor_alphas.feather")
    parser.add_argument("--input_factors_name", default="latent_traits_abs_right.feather")
    parser.add_argument("--input_response_name", default="binary_response_abs_right.feather")
    parser.add_argument("--max_items", default=51, help="after how many items to stop")
    parser.add_argument("--mc_samples", default=500)
    parser.add_argument("--dim", default=6)
    parser.add_argument("--network_version", default="v4")
    parser.add_argument("--pred_type", default="detailed")
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
    # Input Testbank
    arguments = parse_args()
    loadings_dir = pathlib.Path(arguments["cat_cog_models_dir"], arguments["input_params_name"])
    factors_dir = pathlib.Path(arguments["cat_cog_models_dir"], arguments["input_factors_name"])
    response_dir = pathlib.Path(arguments["cat_cog_models_dir"], arguments["input_response_name"])
    loading_df = pd.read_feather(loadings_dir)
    factor_df = pd.read_feather(factors_dir)
    response_df = pd.read_feather(response_dir)
    test_params = {"alphas": loading_df.values[:, :-1], "intercepts": loading_df.values[:, -1],
                   "thetas": factor_df.values[:, :arguments["dim"]]}
    # Input trained NN directory
    file_names = [
                  "online-Q-network-v4-6-factor/NN_bifactor_alphas_150000_first0-1_detailed_v4_1.5e-05_128_30000_600000_0.14/fn_69000.pt",
                 ]

    for filename in file_names:
        print(filename)
        nn_input_dir = pathlib.Path(arguments["cat_cog_models_dir"]).joinpath(filename)
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
        filename = "x08_s05_QCAT_{}_{}_{}_{}_final.pickle".format(arguments["max_items"],filename.split("/")[0], filename.split("/")[1], filename.split("/")[2] )
        model_output_dir = pathlib.Path(arguments["cat_cog_models_dir"]).joinpath("Q-CAT")
        with open(model_output_dir.joinpath(filename), 'wb') as handle:
            pickle.dump(model, handle, pickle.HIGHEST_PROTOCOL)
        del model

