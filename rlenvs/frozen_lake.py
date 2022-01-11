import copy
import logging
import math
from itertools import cycle

import numpy as np
from gym.envs.registration import register
from gym.envs.toy_text.discrete import DiscreteEnv
from gym.envs.toy_text.frozen_lake import MAPS

from .dimension import IntegerDimension
from .environment import TIME_LIMIT_MIN, EnvironmentABC, EnvironmentResponse
from .obs_space import IntegerObsSpaceBuilder

_PERF_LB = 0.0
_SLIP_PROB_MIN_INCL = 0.0
_SLIP_PROB_MAX_EXCL = 1.0
_IOD_STRATS = ("top_left", "frozen_uniform_rand", "frozen_no_repeat",
               "frozen_repeat", "ssa_uniform_rand", "ssa_no_repeat",
               "ssa_repeat", "ssb_uniform_rand", "ssb_no_repeat", "ssb_repeat")
_TOP_LEFT_OBS_RAW = 0
# used when registering envs then overwritten later
_DUMMY_MAX_EP_STEPS = TIME_LIMIT_MIN
_DEFAULT_SEED = 0

MAPS["4x4"] = \
    ["SFFF",
     "FHFH",
     "FFFH",
     "HFFG"]

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


def make_frozen_lake_env(grid_size, slip_prob, iod_strat, seed=_DEFAULT_SEED):
    assert _SLIP_PROB_MIN_INCL <= slip_prob < _SLIP_PROB_MAX_EXCL
    assert iod_strat in _IOD_STRATS
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
    return cls(slip_prob, iod_strat, seed)


class FrozenLakeABC(EnvironmentABC):
    """Changes observations and observation space to be an (x, y) grid instead
    of simple numbered array of cells."""
    def __init__(self, slip_prob, iod_strat, seed):
        is_slippery = slip_prob > 0.0
        super().__init__(env_name=self._GYM_ENV_NAME,
                         time_limit=self._TIME_LIMIT,
                         env_kwargs={"is_slippery": is_slippery},
                         custom_obs_space=None,
                         custom_action_space=None,
                         seed=seed)
        self._x_y_coordinates_obs_space = \
            self._gen_x_y_coordinates_obs_space(self._GRID_SIZE)
        self._x_y_obs_cache = self._make_x_y_obs_cache(self._GRID_SIZE)
        self._slip_prob = slip_prob
        self._alter_transition_func_if_needed(self._slip_prob)
        self._inject_csprob_ns_into_wrapped(self._wrapped_env)

        self._iod_strat = iod_strat
        self._frozen_iter = iter(self._get_nonterminal_states_raw())
        self._frozen_cycler = cycle(self._get_nonterminal_states_raw())

        self._ssa_states_raw = [
            self._convert_x_y_obs_to_raw(x_y_obs)
            for x_y_obs in self._SSA_STATES_X_Y
        ]
        self._ssa_iter = iter(self._ssa_states_raw)
        self._ssa_cycler = cycle(self._ssa_states_raw)

        self._ssb_states_raw = [
            self._convert_x_y_obs_to_raw(x_y_obs)
            for x_y_obs in self._SSB_STATES_X_Y
        ]
        self._ssb_iter = iter(self._ssb_states_raw)
        self._ssb_cycler = cycle(self._ssb_states_raw)

        self._si_size = self._calc_si_size(self._iod_strat)

    def _gen_x_y_coordinates_obs_space(self, grid_size):
        obs_space_builder = IntegerObsSpaceBuilder()
        for name in ("x", "y"):
            obs_space_builder.add_dim(
                IntegerDimension(lower=0, upper=(grid_size - 1), name=name))
        return obs_space_builder.create_space()

    def _make_x_y_obs_cache(self, grid_size):
        x_y_obs_cache = {}
        for raw_obs in range(0, grid_size**2):
            x_y_obs_cache[raw_obs] = \
                self._convert_raw_obs_to_x_y_coordinates(raw_obs)
        return x_y_obs_cache

    def _convert_raw_obs_to_x_y_coordinates(self, raw_obs):
        # raw obs is number indicating idx into flattened grid, where 0 is top
        # left, and flattening is done left to right, top to bottom.
        # x is the column coordinate, y is the row coordinate, both starting
        # from 0.
        x = raw_obs % self._GRID_SIZE
        y = math.floor(raw_obs / self._GRID_SIZE)
        return np.asarray([x, y])

    def _convert_x_y_obs_to_raw(self, x_y_obs):
        (x, y) = x_y_obs
        raw_obs = y * self._GRID_SIZE + x
        assert 0 <= raw_obs <= (self._GRID_SIZE**2 - 1)
        return raw_obs

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

    def _inject_csprob_ns_into_wrapped(self, wrapped_env):
        """Set csprob_ns attribute of wrapped gym DiscreteEnv *after transition
        matrix P has been modified* by this wrapper."""
        d_env = wrapped_env.unwrapped
        assert isinstance(d_env, DiscreteEnv)
        d_env.csprob_ns_with_idxs = d_env.gen_csprob_ns_with_idxs()

    def _get_nonterminal_states_raw(self):
        desc = self._wrapped_env.desc.flatten()
        nonterminal_states_raw = [
            idx for (idx, letter) in enumerate(desc)
            if letter == b'S' or letter == b'F'
        ]
        return nonterminal_states_raw

    def _calc_si_size(self, iod_strat):
        if self._iod_strat == "top_left":
            return 1
        elif "frozen" in self._iod_strat:
            return len(self.nonterminal_states)
        elif "ssa" in self._iod_strat:
            return len(self._ssa_states_raw)
        elif "ssb" in self._iod_strat:
            return len(self._ssb_states_raw)
        else:
            assert False

    @property
    def perf_lower_bound(self):
        return _PERF_LB

    @property
    def si_size(self):
        return self._si_size

    def _sample_initial_obs(self):
        if self._iod_strat == "top_left":
            return _TOP_LEFT_OBS_RAW
        elif self._iod_strat == "frozen_uniform_rand":
            return self._iod_rng.choice(self._get_nonterminal_states_raw())
        elif self._iod_strat == "frozen_no_repeat":
            return next(self._frozen_iter)
        elif self._iod_strat == "frozen_repeat":
            return next(self._frozen_cycler)
        elif self._iod_strat == "ssa_uniform_rand":
            return self._iod_rng.choice(self._ssa_states_raw)
        elif self._iod_strat == "ssa_no_repeat":
            return next(self._ssa_iter)
        elif self._iod_strat == "ssa_repeat":
            return next(self._ssa_cycler)
        elif self._iod_strat == "ssb_uniform_rand":
            return self._iod_rng.choice(self._ssb_states_raw)
        elif self._iod_strat == "ssb_no_repeat":
            return next(self._ssb_iter)
        elif self._iod_strat == "ssb_repeat":
            return next(self._ssb_cycler)
        else:
            assert False

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
        return self._x_y_obs_cache[raw_obs]

    def step(self, action):
        raw_response = super().step(action)
        x_y_obs = self._x_y_obs_cache[raw_response.obs]
        return EnvironmentResponse(obs=x_y_obs,
                                   reward=raw_response.reward,
                                   is_terminal=raw_response.is_terminal,
                                   info=raw_response.info)


class FrozenLake4x4(FrozenLakeABC):
    _GYM_ENV_NAME = "FrozenLake-v0"
    _GRID_SIZE = 4
    _TIME_LIMIT = 150

    _SSA_STATES_X_Y = [(0, 0), (1, 3), (2, 2), (3, 0)]
    _SSB_STATES_X_Y = [(0, 0), (3, 0)]


class FrozenLake8x8(FrozenLakeABC):
    _GYM_ENV_NAME = "FrozenLake8x8-v0"
    _GRID_SIZE = 8
    _TIME_LIMIT = 300

    _SSA_STATES_X_Y = [(0, 0), (0, 4), (0, 7), (1, 2), (2, 6), (4, 3), (4, 7),
                       (5, 1), (7, 0), (7, 3)]
    _SSB_STATES_X_Y = [(0, 0), (0, 7), (4, 7), (7, 0)]


class FrozenLake12x12(FrozenLakeABC):
    _GYM_ENV_NAME = "FrozenLake12x12-v0"
    _GRID_SIZE = 12
    _TIME_LIMIT = 450

    _SSA_STATES_X_Y = [(0, 0), (0, 3), (0, 8), (0, 11), (2, 6), (2, 10),
                       (3, 2), (4, 0), (4, 3), (4, 11), (5, 2), (5, 5), (7, 0),
                       (7, 4), (7, 10), (8, 8), (9, 11), (10, 4), (10, 9),
                       (11, 0), (11, 7)]
    _SSB_STATES_X_Y = [(0, 0), (0, 11), (3, 2), (4, 0), (4, 3), (4, 11),
                       (5, 2), (9, 11), (11, 0)]


class FrozenLake16x16(FrozenLakeABC):
    _GYM_ENV_NAME = "FrozenLake16x16-v0"
    _GRID_SIZE = 16
    _TIME_LIMIT = 600
