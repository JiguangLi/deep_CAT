import numpy as np
import torch
import bayesian_cat as bcat
from torch.nn import MSELoss
import pathlib
from util import EarlyStopping
import pickle
import typing


def epsilon_greedy_policy(state, q_network, epsilon, action_dim):
    x, x_stats, x_preds, mask = state
    avail_items = np.arange(0, action_dim, 1)
    avail_items = avail_items[mask==1]
    if np.random.uniform(0,1) < epsilon:
        return np.random.choice(avail_items)
    else:
        q_network.eval()
        with torch.no_grad():
            q_values = q_network(
                torch.tensor(x, dtype=torch.float32).unsqueeze(0),
                torch.tensor(x_stats, dtype=torch.float32).unsqueeze(0),
                torch.tensor(x_preds, dtype=torch.float32).unsqueeze(0),
                torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            )
        q_network.train()
        return torch.argmax(q_values , dim=1)

def online_train_dqn(
        alphas: np.ndarray,
        intercepts: np.ndarray,
        num_episodes: int,
        learning_rate: float,
        variance_threshold: float,
        target_update_freq: int,
        batch_size: int,
        buffer_size: int,
        learner_eval_sample: int,
        gamma: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decrease_steps: int,
        lr_decrease_patience: int,
        eval_frequency: int,
        early_stopping: int,
        max_item_per_learner: int,
        num_workers: int,
        save_frequency: int,
        save_path: str
):
    """Training Deep adaptive learning models - online prediction version

    @Args:
    target_update_freq: how often to update target network
    variance_threshold: max posterior variance threshold
    epsilon_decrease_steps: how many steps for episilon to decrease to min
    lr_decrease_patience: how many times the average rewards no longer improves should we decease LR
    eval_frequency: how many episodes of rewards should we average to determine whether to decrease LR
                    and early stopping. We need early stopping > lr_decrease_patience typically
    early_stopping: after how many lr_eval_frequency when rewards no longer improves should we early stop
    save_frequency: how many episodes to save the network
    """
    # Set CUDA or CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up parameters
    m, k = alphas.shape
    epsilon_step_size = (epsilon_start - epsilon_end) / epsilon_decrease_steps
    average_rewards, average_loss = 0, 0.0

    # Instantiate model
    input_dim = k + 2
    input_stats_dim = int(k * 4 + (1 + k) * k / 2)
    model = bcat.OnlineQNetwork(
        input_dim=input_dim,
        input_stats_dim=input_stats_dim,
        num_items=m,
        hidden_dim_phi1=256,
    ).to(device)
    target_network = bcat.OnlineQNetwork(
        input_dim=input_dim,
        input_stats_dim=input_stats_dim,
        num_items=m,
        hidden_dim_phi1=256,
    ).to(device)
    target_network.load_state_dict(model.state_dict())
    target_network.eval()

    # set up criterion, optimizer, scheduler(decrease when average rewards no longer improves over episodes), and buffer
    criterion = MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, mode='max',
                                                           patience=lr_decrease_patience,
                                                           verbose=True, min_lr=5 * 1e-7)
    early_stopping = EarlyStopping(early_stopping)
    buffer = bcat.OnlineReplayBuffer(buffer_size)

    # initialize a training log
    log_file_global = pathlib.Path(save_path).joinpath("global_training_log.txt")
    log_global = open(log_file_global, "w")
    detailed_log = {}

    cur_epsilon = epsilon_start
    for episode in range(int(num_episodes)):
        print(episode)
        theta = np.random.normal(0, 1, k)
        episode_learner = bcat.OnlineEpisodeLearner(
            point_theta=theta,
            alphas=alphas,
            intercepts=intercepts,
            num_items=m,
            max_item=max_item_per_learner,
            num_workers=num_workers,
            eval_samples=learner_eval_sample
        )
        learner_reward, learned_loss = 0, 0.0
        cur_state = episode_learner.get_state(0)  # (pos_params, running_stats, mask)
        item_count = 0
        while True:
            # simulation to next state
            action = epsilon_greedy_policy(cur_state, model, cur_epsilon, m)
            episode_learner.answer_item(alphas[action], intercepts[action], action)
            item_count += 1
            reward, done = episode_learner.eval_latent_traits(variance_threshold)
            print(episode, item_count, reward, cur_epsilon)
            new_state = episode_learner.get_state(item_count)
            buffer.add((cur_state, action, reward, new_state, done))
            cur_state = new_state
            learner_reward += reward
            if done or item_count >= max_item_per_learner:
                break
            # update network
            if buffer.size() > batch_size:
                # unpack buffer samples
                mini_batch = buffer.sample(batch_size)
                sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = zip(*mini_batch)
                pos_param, pos_stats, pos_preds, pos_masks = zip(*sampled_states)
                pos_params = torch.tensor(np.array(pos_param), dtype=torch.float32).to(device)
                pos_stats = torch.tensor(np.array(pos_stats), dtype=torch.float32).to(device)
                pos_preds = torch.tensor(np.array(pos_preds), dtype=torch.float32).to(device)
                pos_masks = torch.tensor(np.array(pos_masks), dtype=torch.float32).to(device)
                sampled_actions = torch.LongTensor(sampled_actions).unsqueeze(1)
                sampled_rewards = torch.FloatTensor(sampled_rewards).unsqueeze(1)
                next_pos_param, next_pos_stats, next_pos_preds, next_pos_masks = zip(*sampled_next_states)
                next_pos_params = torch.tensor(np.array(next_pos_param), dtype=torch.float32).to(device)
                next_pos_stats = torch.tensor(np.array(next_pos_stats), dtype=torch.float32).to(device)
                next_pos_preds = torch.tensor(np.array(next_pos_preds), dtype=torch.float32).to(device)
                next_pos_masks = torch.tensor(np.array(next_pos_masks), dtype=torch.float32).to(device)
                dones = torch.FloatTensor(sampled_dones).unsqueeze(1)
                # Compute the target Q-values
                with torch.no_grad():
                    target_q_values = sampled_rewards + (1 - dones) * gamma * target_network(
                        next_pos_params, next_pos_stats, next_pos_preds, next_pos_masks).max(1, keepdim=True)[0]
                # Compute the current Q-values
                q_values = model(pos_params, pos_stats, pos_preds, pos_masks).gather(1, sampled_actions)
                print("q_values")
                print(q_values[:10].flatten())
                print("target_q_values")
                print(target_q_values[:10].flatten())
                # Compute the loss
                loss = criterion(q_values, target_q_values)
                learned_loss += loss.item()
                # Optimize the Q-network
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # decay epsilon
            cur_epsilon = max(epsilon_end, cur_epsilon - epsilon_step_size)

        # end of testing
        arr_str = np.array2string(theta)
        detailed_log[episode] = {"learned_reward": learner_reward, "theta": theta, "learned_loss": learned_loss}
        log_entry = f"Episode {episode}, Learned Reward: {learner_reward}, Learned Loss: {learned_loss}, theta:{arr_str}"
        print(log_entry)
        average_rewards += learner_reward
        average_loss += learned_loss
        if episode % eval_frequency == 0 and episode > 0:
            average_rewards = average_rewards / eval_frequency
            average_loss = average_loss / eval_frequency
            log_entry = f"Episode {episode}, Average Reward {average_rewards}, Average Loss {average_loss}"
            print("################################")
            print(log_entry)
            log_global.write(log_entry + "\n")
            scheduler.step(average_rewards)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Episode {episode}, LR: {current_lr}, Epsilon: {cur_epsilon}")
            print("###############################")
            early_stopping(average_rewards)
            average_rewards = 0
            average_loss = 0.0
            target_network.load_state_dict(model.state_dict())
        # Update the target network
        if episode % target_update_freq == 0:
            target_network.load_state_dict(model.state_dict())
        # save model and detailed_log
        if episode % save_frequency == 0:
            torch.save(model.state_dict(), f"{save_path}/fn_{episode}.pt")
            with open(f"{save_path}/detailed_log.pickle", "wb") as handle:
                pickle.dump(detailed_log, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if early_stopping.early_stop:
            break

    log_global.close()

    torch.save(model.state_dict(), f"{save_path}/fn_final.pt")


def online_train_double_dqn(
        alphas: np.ndarray,
        intercepts: np.ndarray,
        num_episodes: int,
        learning_rate: float,
        variance_threshold: float,
        target_update_freq: int,
        batch_size: int,
        buffer_size: int,
        learner_eval_sample: int,
        gamma: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decrease_steps: int,
        lr_decrease_patience: int,
        eval_frequency: int,
        early_stopping: int,
        max_item_per_learner: int,
        reward_type: str,
        pred_type: str,
        network_version: str,
        retrain_params: typing.Dict,
        factor_generate_params: typing.Dict,
        num_workers: int,
        save_frequency: int,
        save_path: str
):
    """Training Deep adaptive learning models - online prediction version

    @Args:
    target_update_freq: how often to update target network
    variance_threshold: max posterior variance threshold
    epsilon_decrease_steps: how many steps for episilon to decrease to min
    lr_decrease_patience: how many times the average rewards no longer improves should we decease LR
    eval_frequency: how many episodes of rewards should we average to determine whether to decrease LR
                    and early stopping. We need early stopping > lr_decrease_patience typically
    early_stopping: after how many lr_eval_frequency when rewards no longer improves should we early stop
    retrain_params: network_folder_dir, network_name, retrain_version
    factor_generate_params: extreme_std, extreme_freq
    save_frequency: how many episodes to save the network
    """
    # Set CUDA or CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up parameters
    m, k = alphas.shape
    epsilon_step_size = (epsilon_start - epsilon_end) / epsilon_decrease_steps
    average_rewards, average_loss = 0, 0.0

    # Instantiate model
    input_dim = k + 2
    input_stats_dim = int(k * 4 + (1 + k) * k / 2)

    if network_version == "v1":
        model = bcat.OnlineQNetwork(
            input_dim=input_dim,
            input_stats_dim=input_stats_dim,
            num_items=m,
            hidden_dim_phi1=256,
        ).to(device)
        target_network = bcat.OnlineQNetwork(
            input_dim=input_dim,
            input_stats_dim=input_stats_dim,
            num_items=m,
            hidden_dim_phi1=256,
        ).to(device)
    elif network_version == "v2":
        model = bcat.OnlineQNetworkV2(
            input_dim=input_dim,
            input_stats_dim=input_stats_dim,
            num_items=m,
            hidden_dim_phi1=256,
        ).to(device)
        target_network = bcat.OnlineQNetworkV2(
            input_dim=input_dim,
            input_stats_dim=input_stats_dim,
            num_items=m,
            hidden_dim_phi1=256,
        ).to(device)
    elif network_version == "v3":
        model = bcat.OnlineQNetworkV3(
            input_dim=input_dim,
            input_stats_dim=input_stats_dim,
            num_items=m,
            hidden_dim_phi1=256,
        ).to(device)
        target_network = bcat.OnlineQNetworkV3(
            input_dim=input_dim,
            input_stats_dim=input_stats_dim,
            num_items=m,
            hidden_dim_phi1=256,
        ).to(device)
    elif network_version == "v4":
        model = bcat.OnlineQNetworkV4(
            input_dim=input_dim,
            input_stats_dim=input_stats_dim,
            num_items=m,
            hidden_dim_phi1=256,
        ).to(device)
        target_network = bcat.OnlineQNetworkV4(
            input_dim=input_dim,
            input_stats_dim=input_stats_dim,
            num_items=m,
            hidden_dim_phi1=256,
        ).to(device)
    else:
        model = bcat.OnlineQNetworkV5(
            input_dim=input_dim,
            input_stats_dim=input_stats_dim,
            num_items=m,
            hidden_dim_phi1=256,
        ).to(device)
        target_network = bcat.OnlineQNetworkV5(
            input_dim=input_dim,
            input_stats_dim=input_stats_dim,
            num_items=m,
            hidden_dim_phi1=256,
        ).to(device)

    if retrain_params:
        network_dir = pathlib.Path(retrain_params["network_folder_dir"]).joinpath(retrain_params["network_name"])
        model.load_state_dict(torch.load(network_dir))
        target_network.load_state_dict(model.state_dict())
        target_network.eval()
    else:
        target_network.load_state_dict(model.state_dict())
        target_network.eval()

        # set up criterion, optimizer, scheduler(decrease when average rewards no longer improves over episodes), and buffer
    criterion = MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, mode='max',
                                                           patience=lr_decrease_patience,
                                                           verbose=True, min_lr=1e-6)
    early_stopping = EarlyStopping(early_stopping)
    buffer = bcat.OnlineReplayBuffer(buffer_size)

    # initialize a training log
    if retrain_params:
        # buffer_start_idx = retrain_params["network_name"].rindex("fn_") + 3
        # buffer_end_idx = retrain_params["network_name"].rindex(".pt")
        # buffer_num = int(retrain_params["network_name"][buffer_start_idx:buffer_end_idx])
        buffer_dir = pathlib.Path(retrain_params["network_folder_dir"]).joinpath("buffer.pkl")
        buffer.load(buffer_dir)
        retrain_version = retrain_params["retrain_version"]
        log_name = f"retrain_{retrain_version}_global_training_log.txt"
        log_file_global = pathlib.Path(save_path).joinpath(log_name)
    else:
        log_file_global = pathlib.Path(save_path).joinpath("global_training_log.txt")
    log_global = open(log_file_global, "w")
    detailed_log = {}

    # start training
    cur_epsilon = epsilon_start
    for episode in range(int(num_episodes)):
        print(episode)
        if factor_generate_params:
            theta = np.random.normal(0, np.where(np.random.rand() <= factor_generate_params["q"],
                                                 factor_generate_params["std"], 1), k)
        else:
            theta = np.random.normal(0, 1, k)
        episode_learner = bcat.OnlineEpisodeLearner(
            point_theta=theta,
            alphas=alphas,
            intercepts=intercepts,
            num_items=m,
            max_item=max_item_per_learner,
            reward_type=reward_type,
            pred_type=pred_type,
            num_workers=8,
            eval_samples=learner_eval_sample
        )
        learner_reward, learned_loss = 0, 0.0
        cur_state = episode_learner.get_state(0)  # (pos_params, running_stats, mask)
        item_count = 0

        # simulating testing process
        while True:
            # simulation to next state
            action = epsilon_greedy_policy(cur_state, model, cur_epsilon, m)
            episode_learner.answer_item(alphas[action], intercepts[action], action)
            item_count += 1
            reward, done = episode_learner.eval_latent_traits(variance_threshold)
            # print(episode, item_count, reward, cur_epsilon)
            new_state = episode_learner.get_state(item_count)
            buffer.add((cur_state, action, reward, new_state, done))
            cur_state = new_state
            learner_reward += reward
            if done or item_count >= max_item_per_learner:
                break
            # update network
            if buffer.size() > batch_size:
                # unpack buffer samples
                mini_batch = buffer.sample(batch_size)
                sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = zip(*mini_batch)
                pos_param, pos_stats, pos_preds, pos_masks = zip(*sampled_states)
                pos_params = torch.tensor(np.array(pos_param), dtype=torch.float32).to(device)
                pos_stats = torch.tensor(np.array(pos_stats), dtype=torch.float32).to(device)
                pos_preds = torch.tensor(np.array(pos_preds), dtype=torch.float32).to(device)
                pos_masks = torch.tensor(np.array(pos_masks), dtype=torch.float32).to(device)
                sampled_actions = torch.LongTensor(sampled_actions).unsqueeze(1)
                sampled_rewards = torch.FloatTensor(sampled_rewards).unsqueeze(1)
                next_pos_param, next_pos_stats, next_pos_preds, next_pos_masks = zip(*sampled_next_states)
                next_pos_params = torch.tensor(np.array(next_pos_param), dtype=torch.float32).to(device)
                next_pos_stats = torch.tensor(np.array(next_pos_stats), dtype=torch.float32).to(device)
                next_pos_preds = torch.tensor(np.array(next_pos_preds), dtype=torch.float32).to(device)
                next_pos_masks = torch.tensor(np.array(next_pos_masks), dtype=torch.float32).to(device)
                dones = torch.FloatTensor(sampled_dones).unsqueeze(1)
                # Compute the target Q-values
                with torch.no_grad():
                    next_actions = model(next_pos_params, next_pos_stats, next_pos_preds, next_pos_masks).argmax(
                        1, keepdim=True)  # CHANGED: eval_network selects actions
                    next_q_values = target_network(next_pos_params, next_pos_stats, next_pos_preds,
                                                   next_pos_masks).gather(1, next_actions)
                    target_q_values = sampled_rewards + (1 - dones) * gamma * next_q_values

                # Compute the current Q-values
                q_values = model(pos_params, pos_stats, pos_preds, pos_masks).gather(1, sampled_actions)
                if episode % 200 == 0:
                    print("q_values")
                    print(episode, torch.mean(q_values), torch.var(q_values))
                    print("target_q_values")
                    print(episode, torch.mean(target_q_values), torch.var(target_q_values))
                # Compute the loss
                loss = criterion(q_values, target_q_values)
                learned_loss += loss.item()
                # Optimize the Q-network
                optimizer.zero_grad()
                loss.backward()
                # Add gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_norm=1.0)  # CHANGED: added gradient clipping
                optimizer.step()

            # decay epsilon
            cur_epsilon = max(epsilon_end, cur_epsilon - epsilon_step_size)

        # end of testing
        arr_str = np.array2string(theta)
        detailed_log[episode] = {"learned_reward": learner_reward, "theta": theta, "learned_loss": learned_loss,
                                 "lr": optimizer.param_groups[0]["lr"], "epsilon": cur_epsilon}
        log_entry = f"Episode {episode}, Learned Reward: {learner_reward}, Learned Loss: {learned_loss}, theta:{arr_str}"
        print(log_entry)
        average_rewards += learner_reward
        average_loss += learned_loss

        if episode % eval_frequency == 0 and episode > 0:
            average_rewards = average_rewards / eval_frequency
            average_loss = average_loss / eval_frequency
            log_entry = f"Episode {episode}, Average Reward {average_rewards}, Average Loss {average_loss}"
            print("################################")
            print(log_entry)
            log_global.write(log_entry + "\n")
            scheduler.step(average_rewards)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Episode {episode}, LR: {current_lr}, Epsilon: {cur_epsilon}")
            print("###############################")
            early_stopping(average_rewards)
            average_rewards = 0
            average_loss = 0.0
            target_network.load_state_dict(model.state_dict())

        # Update the target network
        if episode % target_update_freq == 0:
            target_network.load_state_dict(model.state_dict())
        # save model and detailed_log
        if episode % save_frequency == 0:
            if retrain_params:
                retrain_network_name = retrain_params["network_name"]
                retrain_version = retrain_params["retrain_version"]
                retrain_model_name = f"retrain_{retrain_network_name}_fn_{episode}.pt"
                torch.save(model.state_dict(), f"{save_path}/{retrain_model_name}")
                with open(f"{save_path}/retrain_{retrain_version}_detailed_log.pickle", "wb") as handle:
                    pickle.dump(detailed_log, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                torch.save(model.state_dict(), f"{save_path}/fn_{episode}.pt")
                with open(f"{save_path}/detailed_log.pickle", "wb") as handle:
                    pickle.dump(detailed_log, handle, protocol=pickle.HIGHEST_PROTOCOL)
            buffer.save(f"{save_path}/buffer.pkl")
        if early_stopping.early_stop:
            break

    log_global.close()
    if retrain_params:
        retrain_version = retrain_params["retrain_version"]
        retrain_model_name = f"{save_path}/retrain_{retrain_version}_fn_final.pt"
        torch.save(model.state_dict(), retrain_model_name)
    else:
        torch.save(model.state_dict(), f"{save_path}/fn_final.pt")









