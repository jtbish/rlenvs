import numpy as np

from .dimension import RealDimension
from .environment import EnvironmentABC
from .normalise import NormaliseWrapper
from .obs_space import RealObsSpaceBuilder

_GYM_ENV_NAME = "MountainCar-v0"
_POS_LOWER = -1.2
_POS_UPPER = 0.5
_VEL_LOWER = -0.07
_VEL_UPPER = 0.07
_LEFT_ACTION = 0
_RIGHT_ACTION = 2
# exclude "1" action (do nothing)
_CUSTOM_ACTION_SPACE = (_LEFT_ACTION, _RIGHT_ACTION)
_PERF_LB = -200
_IOD_STRATS = ("bottom_zero_vel", "uniform")
_TIME_LIMIT = 200
_DEFAULT_SEED = 0


def make_mountain_car_env(iod_strat,
                          normalise=False,
                          seed=_DEFAULT_SEED):
    assert iod_strat in _IOD_STRATS
    mc = MountainCar(iod_strat, seed)
    if normalise:
        return NormaliseWrapper(mc)
    else:
        return mc


class MountainCar(EnvironmentABC):
    def __init__(self, iod_strat, seed):
        custom_obs_space = self._gen_custom_obs_space()
        super().__init__(env_name=_GYM_ENV_NAME,
                         time_limit=_TIME_LIMIT,
                         custom_obs_space=custom_obs_space,
                         custom_action_space=_CUSTOM_ACTION_SPACE,
                         seed=seed)
        self._iod_strat = iod_strat

    @property
    def perf_lower_bound(self):
        return _PERF_LB

    def _sample_initial_obs(self):
        if self._iod_strat == "bottom_zero_vel":
            pos = self._iod_rng.uniform(low=-0.6, high=-0.4)
            vel = 0.0
            return np.asarray([pos, vel])
        elif self._iod_strat == "uniform":
            pos = self._iod_rng.uniform(low=_POS_LOWER, high=_POS_UPPER)
            vel = self._iod_rng.uniform(low=_VEL_LOWER, high=_VEL_UPPER)
            return np.asarray([pos, vel])
        else:
            assert False

    def _gen_custom_obs_space(self):
        obs_space_builder = RealObsSpaceBuilder()
        # order of dims is [pos, vel]
        obs_space_builder.add_dim(RealDimension(_POS_LOWER, _POS_UPPER, "pos"))
        obs_space_builder.add_dim(RealDimension(_VEL_LOWER, _VEL_UPPER, "vel"))
        return obs_space_builder.create_space()
