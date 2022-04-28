import numpy as np
from matplotlib import pyplot as plt
# import seaborn as sns
import glob
import tensorflow as tf


def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X, Y = [], []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Train_AverageReturn':
                Y.append(v.simple_value)
    return X, Y


def smooth(scalars, weight=0.6):
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


settings = {
    "breakout": {
        "env_name": "BreakoutNoFrameskip-v4",
        "DQN": {
            'logdir1': 'dqn/data/dqn_Breakout_BreakoutNoFrameskip-v4_26-04-2022_18-28-12/events*',
            'logdir2': 'dqn/data/dqn_Breakout_BreakoutNoFrameskip-v4_26-04-2022_20-37-15/events*',
            'logdir3': 'dqn/data/dqn_Breakout_BreakoutNoFrameskip-v4_26-04-2022_22-38-25/events*',
        },
        "doubleDQN": {
            'logdir1': 'dqn/data/dqn_BreakoutDouble_BreakoutNoFrameskip-v4_24-04-2022_21-27-54/events*',
            'logdir2': 'dqn/data/dqn_BreakoutDouble_BreakoutNoFrameskip-v4_27-04-2022_00-40-22/events*',
            'logdir3': 'dqn/data/dqn_BreakoutDouble_BreakoutNoFrameskip-v4_27-04-2022_10-59-47/events*',
        }
    },
    "pacman": {
        "env_name": "MsPacman-v0",
        "DQN": {
            'logdir1': 'dqn/data/dqn_Pacman_MsPacman-v0_17-04-2022_11-36-50/events*',
            'logdir2': 'dqn/data/dqn_Pacman_MsPacman-v0_28-04-2022_06-50-35/events*',
            'logdir3': 'dqn/data/dqn_Pacman_MsPacman-v0_28-04-2022_08-33-18/events*',
        },
        "doubleDQN": {
            'logdir1': 'dqn/data/dqn_PacmanDouble_MsPacman-v0_17-04-2022_13-13-09/events*',
            'logdir2': 'dqn/data/dqn_PacmanDouble_MsPacman-v0_23-04-2022_13-01-19/events*',
            'logdir3': 'dqn/data/dqn_PacmanDouble_MsPacman-v0_28-04-2022_10-15-37/events*',
        }
    },
    "pong": {
        "env_name": "PongNoFrameskip-v4",
        "DQN": {
            'logdir1': 'dqn/data/dqn_Pong_PongNoFrameskip-v4_22-04-2022_22-44-29/events*',
            'logdir2': 'dqn/data/dqn_Pong_PongNoFrameskip-v4_27-04-2022_13-04-48/events*',
            'logdir3': 'dqn/data/dqn_Pong_PongNoFrameskip-v4_27-04-2022_14-20-18/events*',
        },
        "doubleDQN": {
            'logdir1': 'dqn/data/dqn_PongDouble_PongNoFrameskip-v4_24-04-2022_12-26-32/events*',
            'logdir2': 'dqn/data/dqn_PongDouble_PongNoFrameskip-v4_27-04-2022_15-37-33/events*',
            'logdir3': 'dqn/data/dqn_PongDouble_PongNoFrameskip-v4_27-04-2022_17-04-56/events*',
        }
    },
    "freeway": {
        "env_name": "FreewayNoFrameskip-v4",
        "DQN": {
            'logdir1': 'dqn/data/dqn_Freeway_FreewayNoFrameskip-v4_23-04-2022_02-02-59/events*',
            'logdir2': 'dqn/data/dqn_Freeway_FreewayNoFrameskip-v4_27-04-2022_18-31-00/events*',
            'logdir3': 'dqn/data/dqn_Freeway_FreewayNoFrameskip-v4_27-04-2022_21-49-49/events*',
        },
        "doubleDQN": {
            'logdir1': 'dqn/data/dqn_FreewayDouble_FreewayNoFrameskip-v4_23-04-2022_05-07-20/events*',
            'logdir2': 'dqn/data/dqn_FreewayDouble_FreewayNoFrameskip-v4_28-04-2022_00-48-51/events*',
            'logdir3': 'dqn/data/dqn_FreewayDouble_FreewayNoFrameskip-v4_28-04-2022_03-50-09/events*',
        }
    },
    "lunar": {
        "env_name": "LunarLander-v3",
        "DQN": {
            'logdir1': 'dqn/data/dqn_Lunar_LunarLander-v3_26-04-2022_11-02-57/events*',
            'logdir2': 'dqn/data/dqn_Lunar_LunarLander-v3_26-04-2022_11-28-27/events*',
            'logdir3': 'dqn/data/dqn_Lunar_LunarLander-v3_26-04-2022_11-52-43/events*',
        },
        "doubleDQN": {
            'logdir1': 'dqn/data/dqn_LunarDouble_LunarLander-v3_26-04-2022_12-19-33/events*',
            'logdir2': 'dqn/data/dqn_LunarDouble_LunarLander-v3_26-04-2022_12-46-29/events*',
            'logdir3': 'dqn/data/dqn_LunarDouble_LunarLander-v3_26-04-2022_13-15-14/events*',
        }
    }
}


if __name__ == '__main__':

    env = "pacman"  # options: pong, breakout, freeway, lunar, pacman
    settings = settings[env]
    
    eventfile1 = glob.glob(settings["DQN"]["logdir1"])[0]
    eventfile2 = glob.glob(settings["DQN"]["logdir2"])[0]
    eventfile3 = glob.glob(settings["DQN"]["logdir3"])[0]

    X, Y1 = get_section_results(eventfile1)
    Y1.insert(0, min(Y1))
    Y1 = smooth(Y1)
    X, Y2 = get_section_results(eventfile2)
    Y2.insert(0, min(Y2))
    Y2 = smooth(Y2)
    X, Y3 = get_section_results(eventfile3)
    Y3.insert(0, min(Y3))
    Y3 = smooth(Y3)
    min_len = min(len(Y1), min(len(Y2), len(Y3)))
    X = X[:min_len]
    Y1, Y2, Y3 = Y1[:min_len], Y2[:min_len], Y3[:min_len]
    Y = np.array([Y1, Y2, Y3])
    
    plt.plot(X, Y.max(0), linewidth=0)
    plt.plot(X, Y.min(0), linewidth=0)
    plt.fill_between(X, Y.max(0), Y.min(0), color='#0000ff22')
    plt.plot(X, Y.mean(0), c='#4b0082', label='Deep Q-Learning')
    
    
    eventfile1 = glob.glob(settings["doubleDQN"]["logdir1"])[0]
    eventfile2 = glob.glob(settings["doubleDQN"]["logdir2"])[0]
    eventfile3 = glob.glob(settings["doubleDQN"]["logdir3"])[0]

    X, Y1 = get_section_results(eventfile1)
    Y1.insert(0, min(Y1))
    Y1 = smooth(Y1)
    X, Y2 = get_section_results(eventfile2)
    Y2.insert(0, min(Y2))
    Y2 = smooth(Y2)
    X, Y3 = get_section_results(eventfile3)
    Y3.insert(0, min(Y3))
    Y3 = smooth(Y3)
    min_len = min(len(Y1), min(len(Y2), len(Y3)))
    X = X[:min_len]
    Y1, Y2, Y3 = Y1[:min_len], Y2[:min_len], Y3[:min_len]
    Y = np.array([Y1, Y2, Y3])
    
    plt.plot(X, Y.max(0), linewidth=0)
    plt.plot(X, Y.min(0), linewidth=0)
    plt.fill_between(X, Y.max(0), Y.min(0), color='#ff000022')
    plt.plot(X, Y.mean(0), c='#ff00ff', label='Double Q-Learning')
    
    
    plt.xlabel('Number of Frames Observed')
    plt.ylabel('Cumulative Episode Reward\nMoving average (100 episodes)')
    plt.title('Environment: '+settings["env_name"])
    plt.legend()
    plt.savefig("dqn/visualizations/"+env+".png")
