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
    parser.add_argument("--input_data_name", default= "s01_500_500_4_1.pickle")
    parser.add_argument("--max_items", default=40, help="after how many items to stop")
    parser.add_argument("--mc_samples", default=1000)
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
    nn_input_dir = pathlib.Path(arguments["q_learning_models_dir"]).joinpath("Q-network")
    folder_name = "NN_s01_500_500_4_1_80000_5e-05_128_15000_810000"
    network_name = "fn_63000.pt"
    nn_input_dir = nn_input_dir.joinpath(folder_name)
    nn_input_dir = nn_input_dir.joinpath(network_name)
    # similuate
    model = bcat.DeepQCAT(
        true_thetas=test_params["thetas"],
        test_bank=test_params,
        nn_model_dir = nn_input_dir,
        max_items=arguments["max_items"],
        mc_samples= arguments["mc_samples"],
        num_workers=arguments["num_workers"],
        random_state=arguments["seed"]
    )
    model.simulate()
    filename = "x05_s02_QCAT_{}_{}_{}.pickle".format(arguments["input_data_name"][:-7], folder_name, network_name)
    model_output_dir = pathlib.Path(arguments["q_learning_models_dir"]).joinpath("Q-CAT")
    with open(model_output_dir.joinpath(filename), 'wb') as handle:
        pickle.dump(model, handle, pickle.HIGHEST_PROTOCOL)

