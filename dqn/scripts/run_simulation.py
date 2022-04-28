import sys
import gym
import time
import torch
import numpy as np

from dqn.infrastructure.dqn_utils import create_atari_q_network,\
    create_lander_q_network, wrap_deepmind, register_custom_envs


settings = {
    "freeway": {
        "env_name": "FreewayNoFrameskip-v4",
        "noob_model_path": "dqn/data/dqn_FreewayDṭouble_FreewayNoFrameskip-v4_23-04-2022_05-07-20/agent_itr_60000.pt",
        "amateur_model_path": "dqn/data/dqn_FreewayDouble_FreewayNoFrameskip-v4_23-04-2022_05-07-20/agent_itr_300000.pt",
        "pro_model_path": "dqn/data/dqn_FreewayDouble_FreewayNoFrameskip-v4_23-04-2022_05-07-20/agent_itr_5350000.pt",
        "network_initializer": create_atari_q_network,
        "sleep_time": 0.01,
    },
    "pacman": {
        "env_name": "MsPacman-v0",
        "noob_model_path": "dqn/data/dqn_PacmanDouble_MsPacman-v0_23-04-2022_13-01-19/agent_itr_60000.pt",
        "amateur_model_path": "dqn/data/dqn_PacmanDouble_MsPacman-v0_23-04-2022_13-01-19/agent_itr_910000.pt",
        "pro_model_path": "dqn/data/dqn_PacmanDouble_MsPacman-v0_23-04-2022_13-01-19/agent_itr_13880000.pt",
        "network_initializer": create_atari_q_network,
        "sleep_time": 0.2,
    },
    "pong": {
        "env_name": "PongNoFrameskip-v4",
        "noob_model_path": "dqn/data/dqn_PongDouble_PongNoFrameskip-v4_24-04-2022_12-26-32/agent_itr_760000.pt",
        "amateur_model_path": "dqn/data/dqn_PongDouble_PongNoFrameskip-v4_24-04-2022_12-26-32/agent_itr_1520000.pt",
        "pro_model_path": "dqn/data/dqn_PongDouble_PongNoFrameskip-v4_24-04-2022_12-26-32/agent_itr_7150000.pt",
        "network_initializer": create_atari_q_network,
        "sleep_time": 0.04,
    },
    "breakout": {
        "env_name": "BreakoutNoFrameskip-v4",
        "noob_model_path": "dqn/data/dqn_BreakoutDouble_BreakoutNoFrameskip-v4_24-04-2022_21-27-54/agent_itr_740000.pt",
        "amateur_model_path": "dqn/data/dqn_BreakoutDouble_BreakoutNoFrameskip-v4_24-04-2022_21-27-54/agent_itr_1500000.pt",
        "pro_model_path": "dqn/data/dqn_BreakoutDouble_BreakoutNoFrameskip-v4_24-04-2022_21-27-54/agent_itr_10670000.pt",
        "network_initializer": create_atari_q_network,
        "sleep_time": 0.04,
    },
    "lunar": {
        "env_name": "LunarLander-v3",
        "network_initializer": create_lander_q_network,
        "noob_model_path": "dqn/data/dqn_LunarDouble_LunarLander-v3_27-04-2022_16-47-17/agent_itr_100000.pt",
        "amateur_model_path": "dqn/data/dqn_LunarDouble_LunarLander-v3_27-04-2022_16-47-17/agent_itr_210000.pt",
        "pro_model_path": "dqn/data/dqn_LunarDouble_LunarLander-v3_27-04-2022_16-47-17/agent_itr_350000.pt",
        "sleep_time": 0.01,
    }
}


env = "lunar"
mode = "pro"  # options: noob, amateur, pro

if env == "lunar":
    register_custom_envs()

settings = settings[env]
env = gym.make(settings["env_name"])
if settings["env_name"] == 'LunarLander-v3':
    q_net = settings["network_initializer"](len(env.observation_space.sample()), env.action_space.n)
else:
    q_net = settings["network_initializer"]((84, 84, 4), env.action_space.n)
    env = wrap_deepmind(env)
q_net.load_state_dict(torch.load(settings[str(mode)+"_model_path"]))

rand = False
total_reward = 0.0
if settings["env_name"] == 'LunarLander-v3':
    obs = env.reset()
    while True:
        env.render()
        if rand:
            ac = env.action_space.sample()
        else:
            ac = q_net(torch.FloatTensor(obs)).detach().cpu().numpy() \
                .squeeze().argmax()
        obs, rew, done, _ = env.step(ac)
        total_reward += rew
        if done:
            break
        time.sleep(settings["sleep_time"])
else:
    obs_arr = [None, None, None, env.reset()]
    while True:
        env.render()
        if obs_arr[0] is None:
            ac = 0
        else:
            obs = np.array([np.moveaxis(np.array(obs_arr).squeeze(), 0, -1)])
            if rand:
                ac = env.action_space.sample()
            else:
                ac = q_net(torch.tensor(obs)).detach().cpu().numpy().squeeze().argmax()
        obs, rew, done, _ = env.step(ac)
        obs_arr = obs_arr[1:]
        obs_arr.append(obs)
        total_reward += rew
        if done:
            break
        time.sleep(settings["sleep_time"])
env.close()
print("Total Reward Obtained =", total_reward)
sys.exit(0)
