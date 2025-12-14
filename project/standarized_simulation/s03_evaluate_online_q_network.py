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
    # x09_s01_mixed_grid_500_150_3_2.pickle
    parser.add_argument("--input_data_name", default="x01_lower_triag_500_150_5_4.pickle")
    parser.add_argument("--max_items", default=50, help="after how many items to stop")
    parser.add_argument("--sir_start", default=1, help="after how many items to do posterior reweighting")
    parser.add_argument("--mc_samples", default=500)
    parser.add_argument("--dim", default=5)
    parser.add_argument("--num_workers", default=8)
    parser.add_argument("--network_version", default="v4")
    parser.add_argument("--pred_type", default="detailed")
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
    #Input trained NN directory
    file_names = ["retrain_retrainv1_NN_x01_lower_triag_500_150_5_4_150000_first3-0-1_detailed_v4_1.5e-05_128_30000_1000_0.16/retrain_fn_34000.pt_fn_38000.pt"]
    
    for filename in file_names:
        print(filename)
        nn_input_dir = pathlib.Path(arguments["q_learning_models_dir"]).joinpath("online-Q-network-v4-5-factor-standarized-first3-triag").joinpath(filename)
        # similuate
        model = bcat.OnlineDeepQCAT(
            true_thetas=test_params["thetas"],
            test_bank=test_params,
            nn_model_dir = nn_input_dir,
            max_items=arguments["max_items"],
            mc_samples= arguments["mc_samples"],
            network_version= arguments["network_version"],
            pred_type= arguments["pred_type"],
            num_workers=arguments["num_workers"],
            random_state=arguments["seed"]
        )
        model.simulate()
        save_filename = "triag_5_s03_QCAT_{}_{}_{}_final.pickle".format(arguments["max_items"],filename.split("/")[0], filename.split("/")[1])
        model_output_dir = pathlib.Path(arguments["q_learning_models_dir"]).joinpath("Q-CAT")
        with open(model_output_dir.joinpath(save_filename), 'wb') as handle:
            pickle.dump(model, handle, pickle.HIGHEST_PROTOCOL)
        del model

