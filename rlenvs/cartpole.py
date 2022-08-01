import numpy as np
from scipy.stats import qmc

from .dimension import RealDimension
from .environment import EnvironmentABC
from .normalise import NormaliseWrapper
from .obs_space import RealObsSpaceBuilder

_GYM_ENV_NAME = "CartPole-v0"

_NUM_OBS_DIMS = 4

# These come from an attempt to figure out the largest reasonable span of these
# dimensions, given that gym defines them as -inf to inf, which for our
# purposes is not useful!
# Basically, two sets of experiments were run, one for both the left and right
# actions. Experiment went as follows:
# For 1 million trials:
# - Init CP env using default initial obs. dist. of [-0.05, 0.05] for all dims
# - While trajectory not over, only pick the current action under examination,
# and record all the obss that were encountered.
# Then, after all data was collected, bounds on the 1st and 3rd index of obss
# were found, i.e. min / max values of cart_vel and pole_vel across both
# actions
# These came out at cart_vel max = 2.197468937556396,
# pole_vel max = 3.336540969818183
# Note these bounds were approximately symmetric around zero s.t. the mins were
# negatives of these.
# Then, we simply just rounded these up to nearest tenth decimal value to
# create upper bounds, i.e:
# 2.197468937556396 -> 2.2
# 3.336540969818183 -> 3.4

_MAX_CART_VEL = 2.2
_MAX_POLE_VEL = 3.4

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

_IOD_STRATS = ("dummy", "center_uniform_rand", "center_no_repeat",
               "cover_sample_uniform_rand", "cover_sample_no_repeat")

_TIME_LIMIT = 200
_DEFAULT_SEED = 0
_PERF_LB = 0

_CENTER_OBS_VAL_HIGH = 0.05
_CENTER_OBS_VAL_LOW = -(_CENTER_OBS_VAL_HIGH)

# Maximal hypervolume symmetrical box around the origin for 100 bins per dim
# value function, that contains only maximum values of 200.
# Empirically determined via brute force search.
# ((20, 79), (0, 99), (16, 83), (32, 67))

_NUM_BINS_HALF_DIM = 50  # i.e. 100/2

_COVER_CART_POS_MULT = ((79 - _NUM_BINS_HALF_DIM + 1) / _NUM_BINS_HALF_DIM)
_COVER_CART_POS_HIGH = (_COVER_CART_POS_MULT * _CART_POS_UPPER)
_COVER_CART_POS_LOW = -(_COVER_CART_POS_HIGH)

_COVER_CART_VEL_MULT = ((99 - _NUM_BINS_HALF_DIM + 1) / _NUM_BINS_HALF_DIM)
_COVER_CART_VEL_HIGH = (_COVER_CART_VEL_MULT * _CART_VEL_UPPER)
_COVER_CART_VEL_LOW = -(_COVER_CART_VEL_HIGH)

_COVER_POLE_ANG_MULT = ((83 - _NUM_BINS_HALF_DIM + 1) / _NUM_BINS_HALF_DIM)
_COVER_POLE_ANG_HIGH = (_COVER_POLE_ANG_MULT * _POLE_ANG_UPPER)
_COVER_POLE_ANG_LOW = -(_COVER_POLE_ANG_HIGH)

_COVER_POLE_VEL_MULT = ((67 - _NUM_BINS_HALF_DIM + 1) / _NUM_BINS_HALF_DIM)
_COVER_POLE_VEL_HIGH = (_COVER_POLE_VEL_MULT * _POLE_VEL_UPPER)
_COVER_POLE_VEL_LOW = -(_COVER_POLE_VEL_HIGH)

NUM_CENTER_SAMPLES = 30
# must be a power of 2
NUM_COVER_SAMPLES = 128


def make_cartpole_env(iod_strat, normalise=False, seed=_DEFAULT_SEED):
    assert iod_strat in _IOD_STRATS
    cp = Cartpole(iod_strat, seed)
    if normalise:
        return NormaliseWrapper(cp)
    else:
        return cp


class Cartpole(EnvironmentABC):
    def __init__(self, iod_strat, seed):
        custom_obs_space = self._gen_custom_obs_space()
        super().__init__(env_name=_GYM_ENV_NAME,
                         time_limit=_TIME_LIMIT,
                         custom_obs_space=custom_obs_space,
                         custom_action_space=None,
                         seed=seed)
        self._iod_strat = iod_strat

        if self._iod_strat == "dummy":
            self._dummy_init_obs = np.zeros(len(custom_obs_space))

        elif self._iod_strat == "center_no_repeat":
            self._center_states_iter = iter(self._gen_center_states())

        elif self._iod_strat == "cover_sample_no_repeat":
            self._cover_sample_states_iter = \
                iter(self._gen_cover_sample_states(seed=seed))

    @property
    def perf_lower_bound(self):
        return _PERF_LB

    def _sample_initial_obs(self):
        if self._iod_strat == "dummy":

            return self._dummy_init_obs

        elif self._iod_strat == "center_uniform_rand":

            return self._iod_rng.uniform(low=_CENTER_OBS_VAL_LOW,
                                         high=_CENTER_OBS_VAL_HIGH,
                                         size=_NUM_OBS_DIMS)

        elif self._iod_strat == "center_no_repeat":

            return next(self._center_states_iter)

        elif self._iod_strat == "cover_sample_uniform_rand":

            cart_pos = self._iod_rng.uniform(low=_COVER_CART_POS_LOW,
                                             high=_COVER_CART_POS_HIGH)
            cart_vel = self._iod_rng.uniform(low=_COVER_CART_VEL_LOW,
                                             high=_COVER_CART_VEL_HIGH)
            pole_ang = self._iod_rng.uniform(low=_COVER_POLE_ANG_LOW,
                                             high=_COVER_POLE_ANG_HIGH)
            pole_vel = self._iod_rng.uniform(low=_COVER_POLE_VEL_LOW,
                                             high=_COVER_POLE_VEL_HIGH)
            return np.asarray([cart_pos, cart_vel, pole_ang, pole_vel])

        elif self._iod_strat == "cover_sample_no_repeat":

            return next(self._cover_sample_states_iter)

        else:
            assert False

    def _gen_custom_obs_space(self):
        obs_space_builder = RealObsSpaceBuilder()
        # order of dims is [cart_pos, cart_vel, pole_ang, pole_vel]
        obs_space_builder.add_dim(
            RealDimension(_CART_POS_LOWER, _CART_POS_UPPER, "cart_pos"))
        obs_space_builder.add_dim(
            RealDimension(_CART_VEL_LOWER, _CART_VEL_UPPER, "cart_vel"))
        obs_space_builder.add_dim(
            RealDimension(_POLE_ANG_LOWER, _POLE_ANG_UPPER, "pole_ang"))
        obs_space_builder.add_dim(
            RealDimension(_POLE_VEL_LOWER, _POLE_VEL_UPPER, "pole_vel"))
        return obs_space_builder.create_space()

    def _gen_center_states(self):
        def _gen_state():
            return self._iod_rng.uniform(low=_CENTER_OBS_VAL_LOW,
                                         high=_CENTER_OBS_VAL_HIGH,
                                         size=_NUM_OBS_DIMS)

        return [_gen_state() for _ in range(NUM_CENTER_SAMPLES)]

    def _gen_cover_sample_states(self, seed):
        sobol_sampler = qmc.Sobol(d=_NUM_OBS_DIMS, scramble=True, seed=seed)
        # n = 2^m
        m = np.log2(NUM_COVER_SAMPLES)
        assert m.is_integer()  # i.e. n is a power of 2
        m = int(m)

        # do sobol sample in unit hypercube then scale it for the cover box
        sobol_sample_unit_hypercube = sobol_sampler.random_base2(m)
        cover_l_bounds = np.asarray([
            _COVER_CART_POS_LOW, _COVER_CART_VEL_LOW, _COVER_POLE_ANG_LOW,
            _COVER_POLE_VEL_LOW
        ])
        cover_u_bounds = np.asarray([
            _COVER_CART_POS_HIGH, _COVER_CART_VEL_HIGH, _COVER_POLE_ANG_HIGH,
            _COVER_POLE_VEL_HIGH
        ])
        sobol_sample_cover = \
            qmc.scale(sobol_sample_unit_hypercube, l_bounds=cover_l_bounds,
                      u_bounds=cover_u_bounds, reverse=False)

        sobol_sample_cover = list(sobol_sample_cover)
        assert len(sobol_sample_cover) == NUM_COVER_SAMPLES
        return sobol_sample_cover
