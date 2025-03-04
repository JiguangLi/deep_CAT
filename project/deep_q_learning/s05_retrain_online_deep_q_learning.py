import pathlib
import yaml
import argparse
from onlinetrain import online_train_double_dqn as online_train_dqn
import typing
import pickle


def parse_args(verbose: bool = False) -> typing.Dict[str, typing.Any]:
    """Get command line arguments, with defaults set for local testing."""
    # parse command line arguments first
    parser = argparse.ArgumentParser("Train Policy Network")
    parser.add_argument("--config_filepath",
                        default=pathlib.Path.home().joinpath(pathlib.Path("bayesian-cat", "config", "config.yaml")))
    parser.add_argument("--test_bank_name", default="x09_s01_mixed_grid_500_200_5_4.pickle")
    parser.add_argument("--num_episodes", type=int, default=150000)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1.5e-5)
    parser.add_argument("--variance_threshold", type=float, default=0.16)
    parser.add_argument("--target_update_freq", type=float, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--buffer_size", type=int, default=30000)
    parser.add_argument("--learner_eval_sample", type=int, default=500)
    parser.add_argument("--reward_type", type=str, default="first3-0-1")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon_start", type=float, default=0.02)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--epsilon_decrease_steps", type=int, default=1000)
    parser.add_argument("--lr_decrease_patience", type=int, default=5)
    parser.add_argument("--eval_frequency", type=int, default=1000)
    parser.add_argument("--early_stopping", type=int, default=50)
    parser.add_argument("--max_item_per_learner", type=int, default=70)
    parser.add_argument("--save_frequency", type=int, default=1000)
    parser.add_argument("--pred_type", type=str, default="detailed")
    parser.add_argument("--network_version", type=str, default="v4")
    parser.add_argument("--retrain_version", type=str, default="retrainv1")
    parser.add_argument("--model_name", type=str, default="online-Q-network-v4-5-factor-standarized-first3")
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
    # read test bank
    data_input_dir = pathlib.Path(arguments["sim_data_dir"], arguments["test_bank_name"])
    with open(data_input_dir, 'rb') as handle:
        test_params = pickle.load(handle)
    # create model save path
    model_name = arguments["model_name"]
    num_episodes = arguments["num_episodes"]
    learning_rate = arguments["learning_rate"]
    batch_size = arguments["batch_size"]
    buffer_size = arguments["buffer_size"]
    train_data_name = arguments["test_bank_name"]
    epsilon_steps = arguments["epsilon_decrease_steps"]
    reward_type = arguments["reward_type"]
    pred_type = arguments["pred_type"]
    network_version = arguments["network_version"]
    var_th = arguments["variance_threshold"]
    retrain_version = arguments["retrain_version"]
    _save_path = (
            pathlib.Path(arguments["q_learning_models_dir"])
            / f"{model_name}"
            / f"retrain_{retrain_version}_NN_{train_data_name[:-7]}_{num_episodes}_{reward_type}_{pred_type}_{network_version}_{learning_rate}_{batch_size}_{buffer_size}_{epsilon_steps}_{var_th}"
    )
    _save_path.mkdir(parents=True, exist_ok=True)
    model_save_path = str(_save_path)
    print(model_save_path)

    # retrain_params
    retrain_params ={
        "network_folder_dir" : "/home/jiguang0/bayesian-cat/models/q_learning/online-Q-network-v4-5-factor-standarized-first3/NN_x09_s01_mixed_grid_500_200_5_4_150000_first3-0-1_detailed_v4_1.5e-05_128_30000_750000_0.16",
        "network_name": "fn_24000.pt",
        "retrain_version": retrain_version
    }
    # factor_generate_params
    factor_generate_params = {}
    #factor_generate_params = {
    #     "std": 1.5,
    #     "q": 0.15
    #
    # }

    #Train the model.
    online_train_dqn(
        alphas = test_params["alphas"] ,
        intercepts= test_params["intercepts"] ,
        num_episodes = num_episodes,
        learning_rate = learning_rate,
        variance_threshold = arguments["variance_threshold"],
        target_update_freq = arguments["target_update_freq"],
        batch_size = batch_size,
        buffer_size = buffer_size,
        learner_eval_sample = arguments["learner_eval_sample"],
        gamma = arguments["gamma"],
        epsilon_start = arguments["epsilon_start"],
        epsilon_end = arguments["epsilon_end"],
        epsilon_decrease_steps = arguments["epsilon_decrease_steps"],
        lr_decrease_patience = arguments["lr_decrease_patience"],
        eval_frequency = arguments["eval_frequency"],
        early_stopping = arguments["early_stopping"],
        max_item_per_learner = arguments["max_item_per_learner"],
        reward_type=arguments["reward_type"],
        pred_type = arguments["pred_type"],
        network_version = arguments["network_version"],
        retrain_params = retrain_params,
        factor_generate_params= factor_generate_params,
        num_workers = arguments["num_workers"],
        save_frequency = arguments["save_frequency"],
        save_path = model_save_path
    )











