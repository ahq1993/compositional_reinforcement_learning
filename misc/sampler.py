import numpy as np
import time
import random
from rllab.misc import logger
import pickle

def rollout(env, policy,sub_level_policies,initial_exploration_done, path_length, render=True, speedup=10,g=2):
    Da = env.action_space.flat_dim
    Do = env.observation_space.flat_dim

    observation = env.reset()
    sub_level_obs = observation
    policy.reset()

    observations = np.zeros((path_length + 1, Do))
    _sub_level_actions = np.zeros((path_length, len(sub_level_policies), Da))
    actions = np.zeros((path_length, Da))
    terminals = np.zeros((path_length, ))
    rewards = np.zeros((path_length, ))
    agent_infos = []
    env_infos = []

    t = 0
    for t in range(path_length):

        if g!=0:
            sub_level_obs =observation[:-g]
        else:
            sub_level_obs =observation
        sub_level_actions=[]
        for i in range(0,len(sub_level_policies)):
            action, _ = sub_level_policies[i].get_action(sub_level_obs)
            sub_level_actions.append(action.reshape(1,-1))
        sub_level_actions=np.stack(sub_level_actions,axis=0)
        sub_level_actions=np.transpose(sub_level_actions,(1,0,2))

        action, agent_info = policy.get_action(observation,sub_level_actions)
        next_obs, reward, terminal, env_info = env.step(action)

        agent_infos.append(agent_info)
        env_infos.append(env_info)

        _sub_level_actions[t] = sub_level_actions[0]
        actions[t] = action
        terminals[t] = terminal
        rewards[t] = reward
        observations[t] = observation

        observation = next_obs

        if render:
            env.render()
            time_step = 0.05
            time.sleep(time_step / speedup)

        if terminal:
            break

    observations[t + 1] = observation

    path = {
        'observations': observations[:t + 1],
        'sub_level_actions': _sub_level_actions[:t + 1],
        'actions': actions[:t + 1],
        'rewards': rewards[:t + 1],
        'terminals': terminals[:t + 1],
        'next_observations': observations[1:t + 2],
        'agent_infos': agent_infos,
        'env_infos': env_infos
    }

    return path


def rollouts(env, policy,sub_level_policies,initial_exploration_done, path_length, n_paths,g):
    paths = [
        rollout(env, policy,sub_level_policies,initial_exploration_done, path_length,g=g)
        for i in range(n_paths)
    ]

    return paths


class Sampler(object):
    def __init__(self, max_path_length, min_pool_size, batch_size):
        self._max_path_length = max_path_length
        self._min_pool_size = min_pool_size
        self._batch_size = batch_size

        self.env = None
        self.policy = None
        self.sub_level_policies = None
        self.pool = None

    def initialize(self, env, policy,sub_level_policies, pool):
        self.env = env
        self.policy = policy
        self.sub_level_policies = sub_level_policies
        self.pool = pool

    def set_policy(self, policy):
        self.policy = policy

    def sample(self,initial_exploration_done):
        raise NotImplementedError

    def batch_ready(self):
        enough_samples = self.pool.size >= self._min_pool_size
        return enough_samples

    def random_batch(self):
        return self.pool.random_batch(self._batch_size)

    def terminate(self):
        self.env.terminate()

    def log_diagnostics(self):
        logger.record_tabular('pool-size', self.pool.size)


class SimpleSampler(Sampler):
    def __init__(self, **kwargs):
        super(SimpleSampler, self).__init__(**kwargs)

        self._path_length = 0
        self._path_return = 0
        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 0
        self._current_observation = None
        self._total_samples = 0

    def sample(self,initial_exploration_done,g):
        if self._current_observation is None:
            self._current_observation = self.env.reset()


        sub_level_actions=[]
        sub_level_probs=[]
        sub_level_obs=self._current_observation

        if g!=0:
            sub_level_obs = self._current_observation[:-g]
        else:
            sub_level_obs = self._current_observation


        for i in range(0,len(self.sub_level_policies)):
            action, _ = self.sub_level_policies[i].get_action(sub_level_obs)
            sub_level_actions.append(action.reshape(1,-1))
        sub_level_actions=np.stack(sub_level_actions,axis=0)
        sub_level_actions=np.transpose(sub_level_actions,(1,0,2))

        for i in range(0,len(self.sub_level_policies)):
            pi= np.exp(self.sub_level_policies[i].log_pis_for(sub_level_obs[None]))
            sub_level_probs.append(pi.reshape(1,-1))
        sub_level_probs=np.stack(sub_level_probs,axis=0)
        sub_level_probs=np.transpose(sub_level_probs,(1,0,2))

      
        action, _ = self.policy.get_action(self._current_observation,sub_level_actions)
        next_observation, reward, terminal, info = self.env.step(action)
        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1
        self.pool.add_sample(
            observation=self._current_observation,
            sub_level_actions=sub_level_actions[0],
            sub_level_probs=sub_level_probs[0],
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation)

        if terminal or self._path_length >= self._max_path_length:
            self.policy.reset()
            self._current_observation = self.env.reset()
            self._path_length = 0
            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return

            self._path_return = 0
            self._n_episodes += 1

        else:
            self._current_observation = next_observation

    def log_diagnostics(self):
        super(SimpleSampler, self).log_diagnostics()
        logger.record_tabular('max-path-return', self._max_path_return)
        logger.record_tabular('last-path-return', self._last_path_return)
        logger.record_tabular('episodes', self._n_episodes)
        logger.record_tabular('total-samples', self._total_samples)

class DummySampler(Sampler):
    def __init__(self, batch_size, max_path_length):
        super(DummySampler, self).__init__(
            max_path_length=max_path_length,
            min_pool_size=0,
            batch_size=batch_size)

    def sample(self):
        pass
