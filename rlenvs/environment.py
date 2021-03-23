import abc
import copy
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

    Supports discrete / continuous obs space, discrete action space.
    Supports giving custom obs space / action space."""
    def __init__(self,
                 env_name,
                 env_kwargs=None,
                 custom_obs_space=None,
                 custom_action_space=None,
                 seed=0):
        self._wrapped_env = self._init_wrapped_env(env_name, env_kwargs, seed)
        self._obs_space = self._gen_obs_space_if_not_given(
            self._wrapped_env, custom_obs_space)
        self._action_space = self._gen_action_space_if_not_given(
            self._wrapped_env, custom_action_space)
        self._is_terminal = True

    @property
    @abc.abstractmethod
    def perf_lower_bound(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def perf_upper_bound(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def max_time_steps(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _sample_initial_obs(self):
        raise NotImplementedError

    @property
    def obs_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._action_space

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
            return self._gen_1d_integer_obs_space(wrapped_env)
        elif isinstance(wrapped_env.observation_space, Box):
            return self._gen_nd_real_obs_space(wrapped_env)
        else:
            raise InvalidSpecError(
                "Unrecognised gym environment obs space type.")

    def _gen_1d_integer_obs_space(self, wrapped_env):
        num_obss = wrapped_env.observation_space.n
        obs_space_builder = ObsSpaceBuilder()
        obs_space_builder.add_dim(
            Dimension(lower=0, upper=(num_obss - 1), name="dim0"))
        return obs_space_builder.create_integer_space()

    def _gen_nd_real_obs_space(self, wrapped_env):
        lower_vector = wrapped_env.observation_space.low
        upper_vector = wrapped_env.observation_space.high
        obs_space_builder = ObsSpaceBuilder()
        for (idx, (lower, upper)) in enumerate(zip(lower_vector,
                                                   upper_vector)):
            obs_space_builder.add_dim(
                Dimension(lower=lower, upper=upper, name=f"dim{idx}"))
        return obs_space_builder.create_real_space()

    def _gen_action_space_if_not_given(self, wrapped_env, custom_action_space):
        if custom_action_space is None:
            return self._gen_action_space(wrapped_env)
        else:
            return custom_action_space

    def _gen_action_space(self, wrapped_env):
        num_actions = wrapped_env.action_space.n
        return tuple(range(num_actions))

    def reset(self):
        self._is_terminal = False
        # reset internals in wrapped env
        self._wrapped_env.reset()
        # then gen an initial obs and inject it into wrapped env
        initial_obs = self._sample_initial_obs()
        initial_obs_valid = self._enforce_valid_obs(initial_obs)
        self._inject_obs_into_wrapped_env(initial_obs_valid)
        return initial_obs_valid

    def _enforce_valid_obs(self, obs):
        obs = np.atleast_1d(obs)
        obs = self._truncate_obs(obs)
        return obs

    def _truncate_obs(self, obs):
        """Necessary to enforce observations are in (possibly) custom obs
        space."""
        truncated_obs = []
        for (feature_val, dim) in zip(obs, self._obs_space):
            feature_val = max(feature_val, dim.lower)
            feature_val = min(feature_val, dim.upper)
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
        assert action in self._action_space
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

    def assess_perf(self,
                    policy,
                    num_rollouts,
                    gamma,
                    returns_agg_func=np.mean):
        assert 0.0 <= gamma <= 1.0
        env = copy.deepcopy(self)
        return _common_env_perf_assessment(env, policy, num_rollouts, gamma,
                                           returns_agg_func)


def _common_env_perf_assessment(env, policy, num_rollouts, gamma,
                                returns_agg_func):
    returns = []
    for _ in range(num_rollouts):
        obs = env.reset()
        return_ = 0.0
        time_step = 0
        while True:
            action = policy.choose_action(obs)
            env_response = env.step(action)
            obs = env_response.obs
            return_ += ((gamma**time_step) * env_response.reward)
            time_step += 1
            if (env_response.is_terminal or time_step == env.max_time_steps):
                break
        returns.append(return_)
    perf = returns_agg_func(returns)
    return perf
