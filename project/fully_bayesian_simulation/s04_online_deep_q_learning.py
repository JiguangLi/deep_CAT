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
    parser.add_argument("--input_test_name", default="x10_s01_mixed_grid_500_150_3_2.pickle")
    parser.add_argument("--input_params_name", default="mcmc_x10_s01_mixed_grid_500_150_3_2.pickle")
    parser.add_argument("--num_episodes", type=int, default=150000)
    parser.add_argument("--learning_rate", type=float, default=1.5e-5)
    parser.add_argument("--variance_threshold", type=float, default=0.16)
    parser.add_argument("--target_update_freq", type=float, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--buffer_size", type=int, default=30000)
    parser.add_argument("--loading_pos_sample", type=int, default=500)
    parser.add_argument("--learner_eval_sample", type=int, default=1)
    parser.add_argument("--reward_type", type=str, default="0-1")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon_start", type=float, default=0.99)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--epsilon_decrease_steps", type=int, default=700000)
    parser.add_argument("--lr_decrease_patience", type=int, default=5)
    parser.add_argument("--eval_frequency", type=int, default=1000)
    parser.add_argument("--early_stopping", type=int, default=50)
    parser.add_argument("--max_item_per_learner", type=int, default=65)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--save_frequency", type=int, default=1000)
    parser.add_argument("--model_name", type=str, default="bayesian-online-Q-network-v4-3-factor")
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
    # data_input_dir = pathlib.Path(arguments["sim_data_dir"], arguments["input_test_name"])
    # with open(data_input_dir, 'rb') as handle:
    #     test_bank = pickle.load(handle)
    mcmc_input_dir = pathlib.Path(arguments["mcmc_models_dir"], arguments["input_params_name"])
    with open(mcmc_input_dir, 'rb') as handle:
        mcmc_params = pickle.load(handle)
    pos_alphas = mcmc_params.params["alphas"][-arguments["loading_pos_sample"]:]
    pos_alphas = - pos_alphas/1.6
    pos_intercepts = mcmc_params.params["intercepts"][-arguments["loading_pos_sample"]:, :, 0]
    pos_intercepts = pos_intercepts/1.6
    # create model save path
    model_name = arguments["model_name"]
    num_episodes = arguments["num_episodes"]
    learning_rate = arguments["learning_rate"]
    batch_size = arguments["batch_size"]
    buffer_size = arguments["buffer_size"]
    train_data_name = arguments["input_params_name"]
    epsilon_steps = arguments["epsilon_decrease_steps"]
    reward_type = arguments["reward_type"]
    variance_threshold = arguments["variance_threshold"]
    _save_path = (
            pathlib.Path(arguments["q_learning_models_dir"])
            / f"{model_name}"
            / f"NN_{train_data_name[:-7]}_{num_episodes}_{reward_type}_{variance_threshold}_{learning_rate}_{batch_size}_{buffer_size}_{epsilon_steps}"
    )
    _save_path.mkdir(parents=True, exist_ok=True)
    model_save_path = str(_save_path)
    print(model_save_path)
    #Train the model.
    online_train_dqn(
        alphas = pos_alphas ,
        intercepts= pos_intercepts ,
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
        retrain_params = {},
        factor_generate_params= {},
        num_workers = arguments["num_workers"],
        save_frequency = arguments["save_frequency"],
        save_path = model_save_path
    )










