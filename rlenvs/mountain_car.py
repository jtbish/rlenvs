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
# "bottom_zero_vel" for backwards compatability
# "cover_grid_uniform_rand" is alias for "uniform_rand"
_IOD_STRATS = ("bottom_zero_vel", "bottom_zero_vel_uniform_rand",
               "bottom_zero_vel_no_repeat", "cover_grid_uniform_rand",
               "cover_grid_no_repeat", "uniform_rand")
_TIME_LIMIT = 200
_DEFAULT_SEED = 0

NUM_BOTTOM_ZERO_VEL_SAMPLES = 30
COVER_GRID_SIZE = 13


def make_mountain_car_env(iod_strat, normalise=False, seed=_DEFAULT_SEED):
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

        if self._iod_strat == "bottom_zero_vel_no_repeat":
            self._bottom_zero_vel_states_iter = \
                iter(self._gen_bottom_zero_vel_states())

        elif self._iod_strat == "cover_grid_no_repeat":
            self._cover_grid_states_iter = \
                iter(self._gen_cover_grid_states())

    @property
    def perf_lower_bound(self):
        return _PERF_LB

    def _gen_bottom_zero_vel_states(self):
        def _gen_state():
            pos = self._iod_rng.uniform(low=-0.6, high=-0.4)
            vel = 0.0
            return np.asarray([pos, vel])

        return [_gen_state() for _ in range(NUM_BOTTOM_ZERO_VEL_SAMPLES)]

    def _gen_cover_grid_states(self):
        pos_compts = np.linspace(start=_POS_LOWER,
                                 stop=_POS_UPPER,
                                 num=(COVER_GRID_SIZE + 1),
                                 endpoint=False)
        pos_compts = list(pos_compts)[1:]
        vel_compts = np.linspace(start=_VEL_LOWER,
                                 stop=_VEL_UPPER,
                                 num=(COVER_GRID_SIZE + 1),
                                 endpoint=False)
        vel_compts = list(vel_compts)[1:]

        res = [
            np.asarray([pos, vel]) for pos in pos_compts for vel in vel_compts
        ]
        assert len(res) == (COVER_GRID_SIZE**2)
        return res

    def _sample_initial_obs(self):
        if self._iod_strat in ("bottom_zero_vel",
                               "bottom_zero_vel_uniform_rand"):

            pos = self._iod_rng.uniform(low=-0.6, high=-0.4)
            vel = 0.0
            return np.asarray([pos, vel])

        elif self._iod_strat == "bottom_zero_vel_no_repeat":

            return next(self._bottom_zero_vel_states_iter)

        elif self._iod_strat in ("cover_grid_uniform_rand", "uniform_rand"):

            pos = self._iod_rng.uniform(low=_POS_LOWER, high=_POS_UPPER)
            vel = self._iod_rng.uniform(low=_VEL_LOWER, high=_VEL_UPPER)
            return np.asarray([pos, vel])

        elif self._iod_strat == "cover_grid_no_repeat":

            return next(self._cover_grid_states_iter)

        else:
            assert False

    def _gen_custom_obs_space(self):
        obs_space_builder = RealObsSpaceBuilder()
        # order of dims is [pos, vel]
        obs_space_builder.add_dim(RealDimension(_POS_LOWER, _POS_UPPER, "pos"))
        obs_space_builder.add_dim(RealDimension(_VEL_LOWER, _VEL_UPPER, "vel"))
        return obs_space_builder.create_space()
