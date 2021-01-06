import abc
from collections import namedtuple

import gym
import numpy as np
from gym.spaces import Box, Discrete

from .dimension import Dimension
from .error import EndOfEpisodeError, InvalidSpecError
from .obs_space import ObsSpaceBuilder

EnvironmentResponse = namedtuple("EnvironmentResponse",
                                 ["obs", "reward", "is_terminal"])


class EnvironmentABC(metaclass=abc.ABCMeta):
    """Wrapper over an OpenAI Gym environment to make interface nicer.

    Supports discrete / continuous obs space, discrete action set.
    Supports giving custom obs space / action set."""
    def __init__(self,
                 env_name,
                 env_kwargs=None,
                 custom_obs_space=None,
                 custom_action_set=None,
                 seed=0):
        self._wrapped_env = self._init_wrapped_env(env_name, env_kwargs, seed)
        self._obs_space = self._gen_obs_space_if_not_given(
            self._wrapped_env, custom_obs_space)
        self._action_set = self._gen_action_set_if_not_given(
            self._wrapped_env, custom_action_set)
        self._is_terminal = True

    @property
    @abc.abstractmethod
    def min_perf(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def max_perf(self):
        raise NotImplementedError

    @abc.abstractmethod
    def assess_perf(self, policy):
        raise NotImplementedError

    @abc.abstractmethod
    def assess_perf_and_get_trajs(self, policy):
        raise NotImplementedError

    @property
    def obs_space(self):
        return self._obs_space

    @property
    def action_set(self):
        return self._action_set

    def _init_wrapped_env(self, env_name, env_kwargs, seed):
        if env_kwargs is None:
            env_kwargs = {}
        wrapped_env = gym.make(env_name, **env_kwargs)
        wrapped_env.seed(seed)
        return wrapped_env

    def _gen_obs_space_if_not_given(self, wrapped_env, custom_obs_space):
        if custom_obs_space is None:
            return self._gen_obs_space(wrapped_env)
        else:
            return custom_obs_space

    def _gen_obs_space(self, wrapped_env):
        if isinstance(wrapped_env.observation_space, Discrete):
            return self._gen_discrete_obs_space(wrapped_env)
        elif isinstance(wrapped_env.observation_space, Box):
            return self._gen_continuous_obs_space(wrapped_env)
        else:
            raise InvalidSpecError(
                "Unrecognised gym environment obs space type.")

    def _gen_discrete_obs_space(self, wrapped_env):
        num_obss = wrapped_env.observation_space.n
        obs_space_builder = ObsSpaceBuilder()
        obs_space_builder.add_dim(Dimension(lower=0, upper=(num_obss - 1)))
        return obs_space_builder.create_space()

    def _gen_continuous_obs_space(self, wrapped_env):
        lower_vector = wrapped_env.observation_space.low
        upper_vector = wrapped_env.observation_space.high
        obs_space_builder = ObsSpaceBuilder()
        for (lower, upper) in zip(lower_vector, upper_vector):
            obs_space_builder.add_dim(Dimension(lower, upper))
        return obs_space_builder.create_space()

    def _gen_action_set_if_not_given(self, wrapped_env, custom_action_set):
        if custom_action_set is None:
            return self._gen_action_set(wrapped_env)
        else:
            return custom_action_set

    def _gen_action_set(self, wrapped_env):
        num_actions = wrapped_env.action_space.n
        return tuple(range(num_actions))

    def reset(self):
        self._is_terminal = False
        wrapped_obs = self._wrapped_env.reset()
        wrapped_obs_valid = self._enforce_valid_obs(wrapped_obs)
        self._inject_obs_into_wrapped_env(wrapped_obs_valid)
        return wrapped_obs_valid

    def _enforce_valid_obs(self, obs):
        obs = np.atleast_1d(obs)
        obs = self._truncate_obs(obs)
        return obs

    def _truncate_obs(self, obs):
        """Necessary to enforce observations are in (possibly) custom obs
        space."""
        truncated_obs = []
        for (feature_val, dimension) in zip(obs, self._obs_space):
            feature_val = max(feature_val, dimension.lower)
            feature_val = min(feature_val, dimension.upper)
            truncated_obs.append(feature_val)
        return np.asarray(truncated_obs)

    def step(self, action):
        if self.is_terminal():
            raise EndOfEpisodeError("Environment is out of data (episode is "
                                    "finished). Call env.reset() to "
                                    "reinitialise for next episode.")
        else:
            return self._step(action)

    def _step(self, action):
        assert action in self._action_set, f"{action}, {self._action_set}"
        # do the transition using wrapped env
        wrapped_obs, wrapped_reward, wrapped_done, wrapped_info = \
            self._wrapped_env.step(action)
        # respect the terminal flag given by wrapped env, but enforce that
        # state of wrapped env is within obs space bounds via
        # injecting it back in after being (possibly) truncated
        self._is_terminal = wrapped_done
        wrapped_obs_valid = self._enforce_valid_obs(wrapped_obs)
        self._inject_obs_into_wrapped_env(wrapped_obs_valid)

        return EnvironmentResponse(obs=wrapped_obs_valid,
                                   reward=wrapped_reward,
                                   is_terminal=self._is_terminal)

    def _inject_obs_into_wrapped_env(self, obs):
        self._wrapped_env.unwrapped.state = obs

    def is_terminal(self):
        return self._is_terminal

    def render(self):
        self._wrapped_env.render()
