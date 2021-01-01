import abc
import copy

import numpy as np

from .dimension import Dimension
from .environment import EnvironmentABC
from .obs_space import ObsSpaceBuilder

_GYM_ENV_NAME = "MountainCar-v0"
_POS_LOWER = -1.2
_POS_UPPER = 0.5
_VEL_LOWER = -0.07
_VEL_UPPER = 0.07
_LEFT_ACTION = 0
_RIGHT_ACTION = 2
# exclude "1" action (do nothing)
_CUSTOM_ACTION_SET = (_LEFT_ACTION, _RIGHT_ACTION)

_ENV_SEED = 0
_NUM_PERF_ROLLOUTS = 30
_MAX_TIME_STEPS = 200


def make_mountain_car_a_env():
    return MountainCarVariantA()


def make_mountain_car_b_env():
    return MountainCarVariantB()


def make_mountain_car_c_env():
    return MountainCarVariantC()


def _assess_mountain_car_env_perf(env, policy, returns_agg_func,
                                  return_trajs):
    """Common MC perf. assessment function.

    Can possibly throw UndefinedMappingError if there is a hole in the policy
    and cannot classify an input vec. In that case, let the caller deal with
    the error."""
    returns = []
    trajs = []
    for _ in range(_NUM_PERF_ROLLOUTS):
        obs = env.reset()
        return_ = 0
        traj = []
        for _ in range(_MAX_TIME_STEPS):
            action = policy.classify(obs)
            traj.append((obs, action))
            env_response = env.step(action)
            obs = env_response.obs
            return_ += env_response.reward
            if env_response.is_terminal:
                break
        returns.append(return_)
        trajs.append(traj)
    perf = returns_agg_func(returns)
    if not return_trajs:
        return perf
    else:
        return (perf, trajs)


class MountainCarABC(EnvironmentABC, metaclass=abc.ABCMeta):
    def __init__(self):
        custom_obs_space = self._gen_custom_obs_space()
        custom_action_set = _CUSTOM_ACTION_SET
        super().__init__(env_name=_GYM_ENV_NAME,
                         custom_obs_space=custom_obs_space,
                         custom_action_set=custom_action_set)
        self._rng = np.random.RandomState(_ENV_SEED)

    @property
    def min_perf(self):
        return self._MIN_PERF

    @property
    def max_perf(self):
        return self._MAX_PERF

    def assess_perf(self, policy):
        # pass in a copy of self so each call to this function has same seq.
        # of init obss, i.e. not changing state of self._rng between successive
        # calls
        return _assess_mountain_car_env_perf(
            env=copy.deepcopy(self),
            policy=policy,
            returns_agg_func=self.returns_agg_func,
            return_trajs=False)

    def assess_perf_and_get_trajs(self, policy):
        return _assess_mountain_car_env_perf(
            env=copy.deepcopy(self),
            policy=policy,
            returns_agg_func=self.returns_agg_func,
            return_trajs=True)

    @property
    @abc.abstractmethod
    def returns_agg_func(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _gen_init_obs(self):
        raise NotImplementedError

    def _gen_custom_obs_space(self):
        obs_space_builder = ObsSpaceBuilder()
        # order of dims is [pos, vel]
        obs_space_builder.add_dim(Dimension(_POS_LOWER, _POS_UPPER, "pos"))
        obs_space_builder.add_dim(Dimension(_VEL_LOWER, _VEL_UPPER, "vel"))
        return obs_space_builder.create_space()

    def reset(self):
        # call super reset to let wrapped gym env properly reset everything in
        # its internals, but ignore the obs that is created since want to
        # generate a custom one
        super().reset()
        obs = self._gen_init_obs()
        obs = self._enforce_valid_obs(obs)
        # inject generated obs into the wrapped gym env
        self._wrapped_env.unwrapped.state = obs
        return obs


class MountainCarVariantA(MountainCarABC):
    """Default init obs distribution: pos ~ U(-0.6, -0.4), vel = 0."""
    _MIN_PERF = -200
    _MAX_PERF = -95

    @property
    def returns_agg_func(self):
        return np.mean

    def _gen_init_obs(self):
        pos = self._rng.uniform(low=-0.6, high=-0.4)
        vel = 0.0
        obs = np.asarray([pos, vel])
        return obs


class MountainCarVariantB(MountainCarABC):
    _MIN_PERF = -200
    _MAX_PERF = -75

    @property
    def returns_agg_func(self):
        return np.mean

    def _gen_init_obs(self):
        pos = self._rng.uniform(low=-1.0, high=0.0)
        vel = 0.0
        obs = np.asarray([pos, vel])
        return obs


class MountainCarVariantC(MountainCarABC):
    _MIN_PERF = -200
    _MAX_PERF = -75

    @property
    def returns_agg_func(self):
        return np.mean

    def _gen_init_obs(self):
        pos = self._rng.uniform(low=-1.0, high=0.0)
        vel = self._rng.uniform(low=-0.03, high=0.03)
        obs = np.asarray([pos, vel])
        return obs
