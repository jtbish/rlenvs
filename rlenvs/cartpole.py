import abc
import copy

import numpy as np

from .dimension import Dimension
from .environment import EnvironmentABC
from .obs_space import ObsSpaceBuilder

_GYM_ENV_NAME = "CartPole-v0"
# These were found via the following procedure:
# Run 1 million trials on raw env only picking left action, recording all
# observations
# Run 1 million trials on raw env only picking right action, recording all
# observations
# From both arrays of observations collected (left and right arrays), calc the
# minimum and maximum values of the cart vel and pole vel features.
# Since ran experiment for so long, the min and max values were almost
# symmetrical around zero.
# Finally take these values and multiply them by a leniency factor of 1.05
_MAX_CART_VEL = 2.1975
_MAX_POLE_VEL = 3.3365
_CART_VEL_LOWER = -(_MAX_CART_VEL)
_CART_VEL_UPPER = _MAX_CART_VEL
_POLE_VEL_LOWER = -(_MAX_POLE_VEL)
_POLE_VEL_UPPER = _MAX_POLE_VEL
_MAX_CART_POS = 2.4  # from gym env source code
_CART_POS_LOWER = -_MAX_CART_POS
_CART_POS_UPPER = _MAX_CART_POS
_MAX_POLE_ANG_RADIANS = 12 * (np.pi / 180)  # from gym env source code
_POLE_ANG_LOWER = -(_MAX_POLE_ANG_RADIANS)
_POLE_ANG_UPPER = _MAX_POLE_ANG_RADIANS

_ENV_SEED = 0
_NUM_PERF_ROLLOUTS = 30
_MAX_TIME_STEPS = 200


def make_cartpole_a_env():
    return CartpoleVariantA()


def _assess_cartpole_env_perf(env, policy, returns_agg_func, return_trajs):
    """Common cartpole perf. assessment function.

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


class CartpoleABC(EnvironmentABC, metaclass=abc.ABCMeta):
    def __init__(self):
        custom_obs_space = self._gen_custom_obs_space()
        super().__init__(env_name=_GYM_ENV_NAME,
                         custom_obs_space=custom_obs_space,
                         custom_action_set=None)
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
        return _assess_cartpole_env_perf(
            env=copy.deepcopy(self),
            policy=policy,
            returns_agg_func=self.returns_agg_func,
            return_trajs=False)

    def assess_perf_and_get_trajs(self, policy):
        return _assess_cartpole_env_perf(
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
        # order of dims is [cart_pos, cart_vel, pole_ang, pole_vel]
        obs_space_builder.add_dim(Dimension(_CART_POS_LOWER, _CART_POS_UPPER))
        obs_space_builder.add_dim(Dimension(_CART_VEL_LOWER, _CART_VEL_UPPER))
        obs_space_builder.add_dim(Dimension(_POLE_ANG_LOWER, _POLE_ANG_UPPER))
        obs_space_builder.add_dim(Dimension(_POLE_VEL_LOWER, _POLE_VEL_UPPER))
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


class CartpoleVariantA(CartpoleABC):
    """Default init obs distribution: all ~ U(-0.05, 0.05)"""
    _MIN_PERF = 0
    _MAX_PERF = 200

    @property
    def returns_agg_func(self):
        return np.mean

    def _gen_init_obs(self):
        return np.asarray(self._rng.uniform(low=-0.05, high=0.05, size=4))
