import pathlib
import yaml
import argparse
from train import train_double_dqn
import typing
import pickle


def parse_args(verbose: bool = False) -> typing.Dict[str, typing.Any]:
    """Get command line arguments, with defaults set for local testing."""
    # parse command line arguments first
    parser = argparse.ArgumentParser("Train Policy Network")
    parser.add_argument("--config_filepath",
                        default=pathlib.Path.home().joinpath(pathlib.Path("bayesian-cat", "config", "config.yaml")))
    parser.add_argument("--test_bank_name", default="s01_500_500_4_1.pickle")
    parser.add_argument("--num_episodes", type=int, default=30000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--variance_threshold", type=float, default=0.2)
    parser.add_argument("--target_update_freq", type=float, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--learner_eval_sample", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon_start", type=float, default=0.99)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--epsilon_decrease_steps", type=int, default=100000)
    parser.add_argument("--lr_decrease_patience", type=int, default=4)
    parser.add_argument("--eval_frequency", type=int, default=500)
    parser.add_argument("--early_stopping", type=int, default=30)
    parser.add_argument("--max_item_per_learner", type=int, default=40)
    parser.add_argument("--save_frequency", type=int, default=500)
    parser.add_argument("--model_name", type=str, default="Q-network")
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
    _save_path = (
            pathlib.Path(arguments["q_learning_models_dir"])
            / f"{model_name}"
            / f"NN_{train_data_name[:-7]}_{num_episodes}_{learning_rate}_{batch_size}_{buffer_size}_{epsilon_steps}"
    )
    _save_path.mkdir(parents=True, exist_ok=True)
    model_save_path = str(_save_path)
    print(model_save_path)
    retrain_params = {}
    factor_generate_params = {}
    #Train the model.
    train_double_dqn(
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
        retrain_params= retrain_params,
        factor_generate_params= factor_generate_params,
        save_frequency = arguments["save_frequency"],
        save_path = model_save_path
    )










