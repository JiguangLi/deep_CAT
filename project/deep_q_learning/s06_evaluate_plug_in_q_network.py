import pathlib
import yaml
import bayesian_cat as bcat
import argparse
import typing
import pickle


def parse_args(verbose: bool = False) -> typing.Dict[str, typing.Any]:
    """Get command line arguments, with defaults set for local testing."""
    # parse command line arguments first
    parser = argparse.ArgumentParser("Deep Q Network for Item selection")
    parser.add_argument("--config_filepath",
                        default= pathlib.Path.home().joinpath(pathlib.Path("bayesian-cat", "config", "config.yaml")))
    parser.add_argument("--verbose", default=True)
    parser.add_argument("--input_data_name", default= "s017_mixed_grid_500_150_3_2.pickle")
    parser.add_argument("--max_items", default=30, help="after how many items to stop")
    parser.add_argument("--mc_samples", default=500)
    parser.add_argument("--gamma", default=0.95)
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
    data_input_dir = pathlib.Path(arguments["sim_data_dir"], arguments["input_data_name"])
    with open(data_input_dir, 'rb') as handle:
        test_params = pickle.load(handle)
    # Input trained NN directory
    folder_name = "NN_s017_mixed_grid_500_150_3_2_150000_mi_detailed_v4_1.2e-05_128_30000_1200000_0.13"
    network_name = "fn_26000.pt"
    nn_input_dir = pathlib.Path(arguments["q_learning_models_dir"]).joinpath("online-Q-network-v4-3factor-mixed-grid-mi-nonpb")
    nn_input_dir = nn_input_dir.joinpath(folder_name)
    nn_input_dir = nn_input_dir.joinpath(network_name)
    # similuate
    model = bcat.PluginOnlineDeepQCATV4(
        true_thetas=test_params["thetas"][:2],
        test_bank=test_params,
        nn_model_dir = nn_input_dir,
        max_items=arguments["max_items"],
        mc_samples= arguments["mc_samples"],
        gamma = arguments["gamma"],
        num_workers=arguments["num_workers"],
        random_state=arguments["seed"]
    )
    model.simulate()
    filename = "x05_s06_QCAT_{}_{}_{}_final.pickle".format(arguments["input_data_name"][:-7], folder_name, network_name)
    model_output_dir = pathlib.Path(arguments["q_learning_models_dir"]).joinpath("Q-CAT")
    with open(model_output_dir.joinpath(filename), 'wb') as handle:
        pickle.dump(model, handle, pickle.HIGHEST_PROTOCOL)

