import abc
import copy
from collections import namedtuple

import gym
import numpy as np
from gym.envs.toy_text.discrete import DiscreteEnv
from gym.spaces import Box, Discrete
from gym.wrappers import TimeLimit

from .dimension import IntegerDimension, RealDimension
from .error import EndOfEpisodeError, InvalidSpecError
from .obs_space import IntegerObsSpaceBuilder, RealObsSpaceBuilder

_GAMMA_MIN = 0.0
_GAMMA_MAX = 1.0
_NUM_ROLLOUTS_MIN = 1
TIME_LIMIT_MIN = 1
NULL_ACTION = -1

EnvironmentResponse = namedtuple("EnvironmentResponse",
                                 ["obs", "reward", "is_terminal", "info"])
PerfAssessmentResponse = namedtuple(
    "PerfAssessmentResponse",
    ["perf", "time_steps_used", "time_limit_trunced", "failed"])


class EnvironmentABC(metaclass=abc.ABCMeta):
    """Wrapper over an OpenAI Gym environment to make interface nicer.

    Supports integer / real obs space, integer action space.
    Supports giving custom obs space / action space."""
    def __init__(self,
                 env_name,
                 time_limit,
                 env_kwargs=None,
                 custom_obs_space=None,
                 custom_action_space=None,
                 seed=0):
        self._wrapped_env = self._init_wrapped_env(env_name, env_kwargs, seed)
        self._set_wrapped_time_limit(time_limit)
        self._time_limit = time_limit
        self._obs_space = self._gen_obs_space_if_not_given(
            self._wrapped_env, custom_obs_space)
        self._do_obs_truncation = (custom_obs_space is not None)
        self._action_space = self._gen_action_space_if_not_given(
            self._wrapped_env, custom_action_space)
        self._iod_rng = self._make_iod_rng(seed)
        self._is_terminal = True

    @property
    @abc.abstractmethod
    def perf_lower_bound(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _sample_initial_obs(self):
        raise NotImplementedError

    @property
    def time_limit(self):
        return self._time_limit

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

    def _set_wrapped_time_limit(self, time_limit):
        assert isinstance(self._wrapped_env, TimeLimit)
        assert time_limit is not None
        assert time_limit >= TIME_LIMIT_MIN
        self._wrapped_env._max_episode_steps = time_limit

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
        obs_space_builder = IntegerObsSpaceBuilder()
        obs_space_builder.add_dim(
            IntegerDimension(lower=0, upper=(num_obss - 1), name="dim0"))
        return obs_space_builder.create_space()

    def _gen_nd_real_obs_space(self, wrapped_env):
        lower_vector = wrapped_env.observation_space.low
        upper_vector = wrapped_env.observation_space.high
        obs_space_builder = RealObsSpaceBuilder()
        for (idx, (lower, upper)) in enumerate(zip(lower_vector,
                                                   upper_vector)):
            obs_space_builder.add_dim(
                RealDimension(lower=lower, upper=upper, name=f"dim{idx}"))
        return obs_space_builder.create_space()

    def _gen_action_space_if_not_given(self, wrapped_env, custom_action_space):
        if custom_action_space is None:
            return self._gen_action_space(wrapped_env)
        else:
            return custom_action_space

    def _gen_action_space(self, wrapped_env):
        num_actions = wrapped_env.action_space.n
        return tuple(range(num_actions))

    def _make_iod_rng(self, seed):
        """Initial Observation Distribution Random Number Generator."""
        return np.random.RandomState(int(seed))

    def reset(self):
        self._is_terminal = False
        # reset internals in wrapped env
        self._wrapped_env.reset()
        # then gen an initial obs and inject it into wrapped env
        initial_obs = self._sample_initial_obs()
        self._inject_obs_into_wrapped_env(initial_obs)
        return initial_obs

    def _truncate_obs(self, obs):
        """Necessary to enforce observations are in (possibly) custom obs
        space."""
        if np.isscalar(obs):
            assert len(self._obs_space) == 1
            dim = self._obs_space.dims[0]
            return self._clip_obs_compt(obs_compt=obs, dim=dim)
        else:
            truncated_obs = []
            for (obs_compt, dim) in zip(obs, self._obs_space):
                obs_compt = self._clip_obs_compt(obs_compt, dim)
                truncated_obs.append(obs_compt)
            return np.asarray(truncated_obs)

    def _clip_obs_compt(self, obs_compt, dim):
        obs_compt = max(obs_compt, dim.lower)
        obs_compt = min(obs_compt, dim.upper)
        return obs_compt

    def step(self, action):
        if self._is_terminal:
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
        if self._do_obs_truncation:
            wrapped_obs = self._truncate_obs(wrapped_obs)
            self._inject_obs_into_wrapped_env(wrapped_obs)

        return EnvironmentResponse(obs=wrapped_obs,
                                   reward=wrapped_reward,
                                   is_terminal=self._is_terminal,
                                   info=wrapped_info)

    def _inject_obs_into_wrapped_env(self, obs):
        unwrapped_env = self._wrapped_env.unwrapped
        if isinstance(unwrapped_env, DiscreteEnv):
            assert np.isscalar(obs)
            unwrapped_env.s = obs
        else:
            unwrapped_env.state = obs

    def is_terminal(self):
        return self._is_terminal

    def render(self):
        self._wrapped_env.render()

    def reseed_iod_rng(self, new_seed):
        self._iod_rng = self._make_iod_rng(new_seed)

    def reseed_wrapped_rng(self, new_seed):
        self._wrapped_env.seed(new_seed)


def assess_perf(env, policy, num_rollouts, gamma):
    assert _GAMMA_MIN <= gamma <= _GAMMA_MAX
    assert num_rollouts >= _NUM_ROLLOUTS_MIN
    # make copy of env for perf assessment so rng state is not modified
    # across assessments
    env = copy.deepcopy(env)
    return _assess_perf(env, policy, num_rollouts, gamma)


def _assess_perf(env, policy, num_rollouts, gamma):
    time_steps_used = 0
    returns = []
    time_limit_trunced = False

    for _ in range(num_rollouts):
        obs = env.reset()
        return_ = 0.0
        time_step = 0
        while True:
            action = policy.select_action(obs)
            if action == NULL_ACTION:
                return PerfAssessmentResponse(
                    perf=env.perf_lower_bound,
                    time_steps_used=time_steps_used,
                    time_limit_trunced=time_limit_trunced,
                    failed=True)
            env_response = env.step(action)
            time_limit_trunced = time_limit_trunced or \
                env_response.info.get("TimeLimit.truncated", False)
            obs = env_response.obs
            return_ += ((gamma**time_step) * env_response.reward)
            time_step += 1
            time_steps_used += 1
            if env_response.is_terminal:
                break
        returns.append(return_)

    expected_return = np.mean(returns)
    return PerfAssessmentResponse(perf=expected_return,
                                  time_steps_used=time_steps_used,
                                  time_limit_trunced=time_limit_trunced,
                                  failed=False)
