import numpy as np

from dqn.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer
from dqn.policies.argmax_policy import ArgMaxPolicy
from dqn.critics.dqn_critic import DQNCritic


class DQNAgent(object):
    def __init__(self, env, agent_params):

        self.env = env
        self.agent_params = agent_params
        self.batch_size = agent_params['batch_size']
        # import ipdb; ipdb.set_trace()
        self.last_obs = self.env.reset()

        self.num_actions = agent_params['ac_dim']
        self.learning_starts = agent_params['learning_starts']
        self.learning_freq = agent_params['learning_freq']
        self.target_update_freq = agent_params['target_update_freq']

        self.replay_buffer_idx = None
        self.exploration = agent_params['exploration_schedule']
        self.optimizer_spec = agent_params['optimizer_spec']

        self.critic = DQNCritic(agent_params, self.optimizer_spec)
        self.actor = ArgMaxPolicy(self.critic)

        lander = agent_params['env_name'].startswith('LunarLander')
        self.replay_buffer = MemoryOptimizedReplayBuffer(
            agent_params['replay_buffer_size'], agent_params['frame_history_len'], lander=lander)
        self.t = 0
        self.num_param_updates = 0

    def add_to_replay_buffer(self, paths):
        pass

    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """        

        # storing the latest observation ("frame") into the replay buffer
        # the replay buffer used here is `MemoryOptimizedReplayBuffer` in dqn_utils.py
        self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)

        eps = self.exploration.value(self.t)

        # using epsilon greedy exploration when selecting action
        perform_random_action = np.random.random() < eps or self.t < self.learning_starts
        if perform_random_action:
            # taking random action with probability eps 'OR'
            # if the current step number is less that self.learning_starts
            action = self.env.action_space.sample()
        else:
            # The actor will take in multiple previous observations in order to deal with the
            # partial observability of the environment. The most recent `frame_history_len`
            # number of observations is used as input to the actor. 
            frames = self.replay_buffer.encode_recent_observation()
            action = self.actor.get_action(frames)
        
        # Taking a step in the environment using the action from the policy
        self.last_obs, reward, done, info = self.env.step(action)

        # Storing the result of taking this action into the replay buffer
        self.replay_buffer.store_effect(
            self.replay_buffer_idx,
            action,
            reward,
            done
        )

        # If taking this step resulted in done, resetting the env (and the latest observation)
        if done:
            self.last_obs = self.env.reset()

    def sample(self, batch_size):
        if self.replay_buffer.can_sample(self.batch_size):
            return self.replay_buffer.sample(batch_size)
        else:
            return [], [], [], [], []

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = {}
        if (self.t > self.learning_starts
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
        ):

            log = self.critic.update(
                ob_no,
                ac_na,
                next_ob_no,
                re_n,
                terminal_n,
            )

            # Updating the target network periodically 
            if self.num_param_updates % self.target_update_freq == 0:
                self.critic.update_target_network()

            self.num_param_updates += 1

        self.t += 1
        return log

    def save(self, path):
        self.critic.save(path)
