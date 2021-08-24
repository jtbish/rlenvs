import copy
import logging
import math
from itertools import cycle

import numpy as np
from gym.envs.registration import register
from gym.envs.toy_text.frozen_lake import MAPS

from .dimension import Dimension
from .environment import TIME_LIMIT_MIN, EnvironmentABC, EnvironmentResponse
from .obs_space import ObsSpaceBuilder

_PERF_LB = 0.0
_SLIP_PROB_MIN_INCL = 0.0
_SLIP_PROB_MAX_EXCL = 1.0
_IOD_STRATS = ("top_left", "uniform_rand", "frozen_no_repeat", "frozen_repeat",
               "frozen_corners_no_repeat", "frozen_corners_repeat",
               "repr_no_repeat", "repr_repeat")
_TOP_LEFT_OBS_RAW = 0
# used when registering envs then overwritten later
_DUMMY_MAX_EP_STEPS = TIME_LIMIT_MIN
_MIN_TIME_LIMIT_MULT = 3
_DEFAULT_TIME_LIMIT_MULT = 25
_DEFAULT_SEED = 0

# keys are (grid_size, slip_prob) tuples.
# based on rollouts from *frozen corner states*,
# 1 rollout per state for determinstic, 30 for stochastic
_KNOWN_TIME_LIMIT_MULTS = {
    (4, 0.0): 3,
    (4, 0.1): 3,
    (4, 0.25): 4,
    (4, 0.3): 4,
    (4, 0.4): 20,
    (4, 0.5): 14,
    (8, 0.0): 3,
    (8, 0.1): 3,
    (8, 0.25): 5,
    (8, 0.3): 6,
    (8, 0.4): 11,
    (8, 0.5): 20,
    (12, 0.0): 3,
    (12, 0.1): 3,
    (12, 0.25): 4,
    (12, 0.3): 6,
    (12, 0.4): 7,
    (12, 0.5): 6,
    (16, 0.0): 3,
    (16, 0.1): 3,
    (16, 0.25): 4,
    (16, 0.3): 5,
    (16, 0.4): 8,
    (16, 0.5): 10
}

# modify 4x4 map to make bottom left corner F, shift H that was there one cell
# to right
MAPS["4x4"] = \
    ["SFFF",
     "FHFH",
     "FFFH",
     "FHFG"]

MAPS["8x8"] = \
    ["SFFFFFFF",
     "FFFFFFFF",
     "FFFHFFFF",
     "FFFFFHFF",
     "FFFHFFFF",
     "FHHFFFHF",
     "FHFFHFHF",
     "FFFHFFFG"]

MAPS["12x12"] = \
    ["SFFHFFFFFHFF",
     "HFFFHHFFFHFF",
     "FFHFHFFFFFFF",
     "FFFHFHFFHFFF",
     "FHFFFFHFHFFF",
     "FFFFFFFFFHFF",
     "FHFFFHFFFHFF",
     "FFFFFFFFFFHF",
     "FHFFFHFFFFFF",
     "FFFFFHFFFHFF",
     "FFFFHHFFHHFF",
     "FFFHFFFFHFFG"]

MAPS["16x16"] = \
    ["SFHFFFFHFHFFFFFF",
     "FFFFFFFFFHFFHFFH",
     "FFFFFFHFFHFFFHFH",
     "FFFFHHFFHFFFFFFF",
     "FFFFFFFHHFFHFFFF",
     "FFFFFFFHHFHFHFFH",
     "FHFFFHHFFFFFFFFF",
     "FHFFHFFFFFFHHFFF",
     "FFFFFFFFFFFFFFFH",
     "FFFFFFFFFFHFHHFF",
     "FFHHFHFFFFFFFFFF",
     "FFFFHFFFFFFFFHFF",
     "FFFHHHFFFHFFFFFH",
     "FFFFHFFFFFFFFHFF",
     "HFFFHFFHFHFFFHFF",
     "FFFFHHFFFFFFFFHG"]

register(id="FrozenLake12x12-v0",
         entry_point="gym.envs.toy_text:FrozenLakeEnv",
         kwargs={"map_name": "12x12"},
         max_episode_steps=_DUMMY_MAX_EP_STEPS)

register(id="FrozenLake16x16-v0",
         entry_point="gym.envs.toy_text:FrozenLakeEnv",
         kwargs={"map_name": "16x16"},
         max_episode_steps=_DUMMY_MAX_EP_STEPS)


def make_frozen_lake_env(grid_size,
                         slip_prob,
                         iod_strat,
                         time_limit_mult=None,
                         seed=_DEFAULT_SEED):
    assert _SLIP_PROB_MIN_INCL <= slip_prob < _SLIP_PROB_MAX_EXCL
    assert iod_strat in _IOD_STRATS
    # try lookup time limit mult from known ones if not already given
    if time_limit_mult is None:
        time_limit_mult = _KNOWN_TIME_LIMIT_MULTS.get((grid_size, slip_prob),
                                                      None)
    # if time limit mult still null, assign default val
    if time_limit_mult is None:
        logging.warning(f"Assigning FrozenLake time limit mult its default "
                        f"value of {_DEFAULT_TIME_LIMIT_MULT}. This may not "
                        f"be desired behaviour!")
        time_limit_mult = _DEFAULT_TIME_LIMIT_MULT
    assert isinstance(time_limit_mult, int)
    assert time_limit_mult >= _MIN_TIME_LIMIT_MULT
    logging.info(f"Using FrozenLake tl mult of {time_limit_mult} "
                 f"for (gs, sp): ({grid_size}, {slip_prob})")
    if grid_size == 4:
        cls = FrozenLake4x4
    elif grid_size == 8:
        cls = FrozenLake8x8
    elif grid_size == 12:
        cls = FrozenLake12x12
    elif grid_size == 16:
        cls = FrozenLake16x16
    else:
        assert False
    return cls(slip_prob, iod_strat, time_limit_mult, seed)


class FrozenLakeABC(EnvironmentABC):
    """Changes observations and observation space to be an (x, y) grid instead
    of simple numbered array of cells."""
    def __init__(self, slip_prob, iod_strat, time_limit_mult, seed):
        is_slippery = slip_prob > 0.0
        # tl = c_tl * N
        time_limit = time_limit_mult * self._GRID_SIZE
        super().__init__(env_name=self._GYM_ENV_NAME,
                         time_limit=time_limit,
                         env_kwargs={"is_slippery": is_slippery},
                         custom_obs_space=None,
                         custom_action_space=None,
                         seed=seed)
        self._x_y_coordinates_obs_space = \
            self._gen_x_y_coordinates_obs_space(self._GRID_SIZE)
        self._slip_prob = slip_prob
        self._alter_transition_func_if_needed(self._slip_prob)
        self._iod_strat = iod_strat
        self._frozen_iter = self._make_frozen_raw_state_iter()
        self._frozen_cycler = self._make_frozen_raw_state_cycler()
        self._frozen_corners_iter = self._make_frozen_corners_raw_state_iter()
        self._frozen_corners_cycler = \
            self._make_frozen_corners_raw_state_cycler()
        self._repr_states_iter = self._make_repr_raw_state_iter()
        self._repr_states_cycler = self._make_repr_raw_state_cycler()
        self._num_nonterminal_states = len(self.nonterminal_states)

    def _gen_x_y_coordinates_obs_space(self, grid_size):
        obs_space_builder = ObsSpaceBuilder()
        for name in ("x", "y"):
            obs_space_builder.add_dim(
                Dimension(lower=0, upper=(grid_size - 1), name=name))
        return obs_space_builder.create_integer_space()

    def _alter_transition_func_if_needed(self, slip_prob):
        if slip_prob > 0.0:
            self._alter_transition_func(slip_prob)

    def _alter_transition_func(self, slip_prob):
        assert 0.0 < slip_prob <= 1.0
        P = self._wrapped_env.P
        P_mut = copy.deepcopy(P)
        for state in range(self._wrapped_env.nS):
            for action in range(self._wrapped_env.nA):
                P_cell_raw = P[state][action]
                is_slippery_transition = len(P_cell_raw) == 3
                if is_slippery_transition:
                    # middle tuple is the desired location, first
                    # and last are non-desired locations
                    (_, ns_1, r_1, done_1) = P_cell_raw[0]
                    (_, ns_2, r_2, done_2) = P_cell_raw[1]
                    (_, ns_3, r_3, done_3) = P_cell_raw[2]
                    prob_non_desired = slip_prob / 2
                    prob_desired = (1 - slip_prob)
                    P_cell_mut = []
                    if prob_non_desired != 0.0:
                        P_cell_mut.append(
                            (prob_non_desired, ns_1, r_1, done_1))
                        P_cell_mut.append((prob_desired, ns_2, r_2, done_2))
                        P_cell_mut.append(
                            (prob_non_desired, ns_3, r_3, done_3))
                    else:
                        P_cell_mut.append((prob_desired, ns_2, r_2, done_2))
                else:
                    P_cell_mut = P_cell_raw
                P_mut[state][action] = P_cell_mut
        self._wrapped_env.unwrapped.P = P_mut

    def _make_frozen_raw_state_iter(self):
        return iter(self._get_nonterminal_states_raw())

    def _make_frozen_raw_state_cycler(self):
        return cycle(self._get_nonterminal_states_raw())

    def _make_frozen_corners_raw_state_iter(self):
        return iter(self._get_frozen_corners_raw())

    def _make_frozen_corners_raw_state_cycler(self):
        return cycle(self._get_frozen_corners_raw())

    def _make_repr_raw_state_iter(self):
        return iter(self._get_repr_states_raw())

    def _make_repr_raw_state_cycler(self):
        return cycle(self._get_repr_states_raw())

    def _get_nonterminal_states_raw(self):
        desc = self._wrapped_env.desc.flatten()
        nonterminal_states_raw = [
            idx for (idx, letter) in enumerate(desc)
            if letter == b'S' or letter == b'F'
        ]
        return nonterminal_states_raw

    def _get_frozen_corners_raw(self):
        frozen_corners_raw = [0]  # top left
        frozen_corners_raw.append(self._GRID_SIZE - 1)  # top right
        # bottom left
        frozen_corners_raw.append(self._GRID_SIZE * (self._GRID_SIZE - 1))
        return frozen_corners_raw

    def _get_repr_states_raw(self):
        """So called "representative" states are the 3 frozen corners + any state
        in the grid that is "boxed in": i.e. 3/4 of its neighbours are walls or
        holes."""
        return self._get_frozen_corners_raw() + self._BOXED_IN_STATES_RAW

    @property
    def perf_lower_bound(self):
        return _PERF_LB

    def _sample_initial_obs(self):
        if self._iod_strat == "top_left":
            return _TOP_LEFT_OBS_RAW
        elif self._iod_strat == "uniform_rand":
            return self._uniform_random_initial_obs_raw()
        elif self._iod_strat == "frozen_no_repeat":
            return next(self._frozen_iter)
        elif self._iod_strat == "frozen_repeat":
            return next(self._frozen_cycler)
        elif self._iod_strat == "frozen_corners_no_repeat":
            return next(self._frozen_corners_iter)
        elif self._iod_strat == "frozen_corners_repeat":
            return next(self._frozen_corners_cycler)
        elif self._iod_strat == "repr_no_repeat":
            return next(self._repr_states_iter)
        elif self._iod_strat == "repr_repeat":
            return next(self._repr_states_cycler)
        else:
            assert False

    def _uniform_random_initial_obs_raw(self):
        return self._iod_rng.choice(self._get_nonterminal_states_raw())

    @property
    def obs_space(self):
        return self._x_y_coordinates_obs_space

    @property
    def P(self):
        """Transition matrix."""
        return self._wrapped_env.P

    @property
    def grid_size(self):
        return self._GRID_SIZE

    @property
    def slip_prob(self):
        return self._slip_prob

    @property
    def terminal_states(self):
        desc = self._wrapped_env.desc.flatten()
        terminal_states = [
            self._convert_raw_obs_to_x_y_coordinates(idx)
            for (idx, letter) in enumerate(desc)
            if letter == b'H' or letter == b'G'
        ]
        return terminal_states

    @property
    def nonterminal_states(self):
        desc = self._wrapped_env.desc.flatten()
        nonterminal_states = [
            self._convert_raw_obs_to_x_y_coordinates(idx)
            for (idx, letter) in enumerate(desc)
            if letter == b'S' or letter == b'F'
        ]
        return nonterminal_states

    def reset(self):
        raw_obs = super().reset()
        return self._convert_raw_obs_to_x_y_coordinates(raw_obs)

    def _convert_raw_obs_to_x_y_coordinates(self, raw_obs):
        # raw obs is number indicating idx into flattened grid, where 0 is top
        # left, and flattening is done left to right, top to bottom.
        # x is the column coordinate, y is the row coordinate, both starting
        # from 0.
        assert np.isscalar(raw_obs)
        x = raw_obs % self._GRID_SIZE
        y = math.floor(raw_obs / self._GRID_SIZE)
        assert (y * self._GRID_SIZE + x) == raw_obs
        return np.asarray([x, y])

    def step(self, action):
        raw_response = super().step(action)
        raw_obs = raw_response.obs
        converted_obs = self._convert_raw_obs_to_x_y_coordinates(raw_obs)
        return EnvironmentResponse(obs=converted_obs,
                                   reward=raw_response.reward,
                                   is_terminal=raw_response.is_terminal,
                                   info=raw_response.info)


class FrozenLake4x4(FrozenLakeABC):
    _GYM_ENV_NAME = "FrozenLake-v0"
    _GRID_SIZE = 4
    _BOXED_IN_STATES_RAW = []  # no extra boxed in states for 4x4


class FrozenLake8x8(FrozenLakeABC):
    _GYM_ENV_NAME = "FrozenLake8x8-v0"
    _GRID_SIZE = 8
    _BOXED_IN_STATES_RAW = [60]


class FrozenLake12x12(FrozenLakeABC):
    _GYM_ENV_NAME = "FrozenLake12x12-v0"
    _GRID_SIZE = 12
    _BOXED_IN_STATES_RAW = [4, 27, 29, 40, 136, 141]


class FrozenLake16x16(FrozenLakeABC):
    _GYM_ENV_NAME = "FrozenLake16x16-v0"
    _GRID_SIZE = 16
    _BOXED_IN_STATES_RAW = [8, 91, 164, 179, 181]
