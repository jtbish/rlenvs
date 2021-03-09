import copy
import math

import numpy as np

from gym.envs.registration import register
from gym.envs.toy_text.frozen_lake import MAPS

from .dimension import Dimension
from .environment import EnvironmentResponse, EnvironmentABC
from .obs_space import ObsSpaceBuilder

MAPS["16x16"] = \
    ["SFFFFHFFFFFFFFFF",
     "FFFHHFFHHFFFFFFF",
     "FHHFFHFFFFFHFFFF",
     "FFFFFFFFFFFFFFHH",
     "FFFFFFFFHFFHHHFH",
     "FFFFFHHHFFFHFHHH",
     "HFFFFHHFHFHHFFFF",
     "HFHFFFHFHFFFFFFF",
     "FFHFHFFFFFFFFHHF",
     "FFFFFFHFFFFFFFFF",
     "FFFFFFFFFFFFFFFF",
     "FFFFHFFFFFFFHFFF",
     "FHFFFFFFHHFHFFFH",
     "FFFFFHFFFFFFFFFF",
     "FFFFFFFFFFFFHHFF",
     "FFHFHFFFFFFFHHFG"]

register(id="FrozenLake16x16-v0",
         entry_point="gym.envs.toy_text:FrozenLakeEnv",
         kwargs={"map_name": "16x16"})


def make_frozen_lake_4x4_env(slip_prob=0.0, seed=0):
    return FrozenLake4x4(slip_prob, seed)


def make_frozen_lake_8x8_env(slip_prob=0.0, seed=0):
    return FrozenLake8x8(slip_prob, seed)


def make_frozen_lake_16x16_env(slip_prob=0.0, seed=0):
    return FrozenLake16x16(slip_prob, seed)


class FrozenLakeABC(EnvironmentABC):
    """Changes observations and observation space to be an (x, y) grid instead
    of simple numbered array of cells."""
    def __init__(self, slip_prob, seed):
        is_slippery = slip_prob > 0.0
        super().__init__(env_name=self._GYM_ENV_NAME,
                         env_kwargs={"is_slippery": is_slippery},
                         custom_obs_space=None,
                         custom_action_set=None,
                         seed=seed)
        self._x_y_coordinates_obs_space = \
            self._gen_x_y_coordinates_obs_space(self._GRID_SIZE)
        self._slip_prob = slip_prob
        self._alter_transition_func_if_needed(self._slip_prob)

    def _gen_x_y_coordinates_obs_space(self, grid_size):
        obs_space_builder = ObsSpaceBuilder()
        for name in ("x", "y"):
            obs_space_builder.add_dim(Dimension(0, grid_size - 1, name))
        return obs_space_builder.create_space()

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

    @property
    def obs_space(self):
        return self._x_y_coordinates_obs_space

    @property
    def P(self):
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
        terminal_states = [self._convert_raw_obs_to_x_y_coordinates([idx]) for
                           (idx, letter) in enumerate(desc) if letter == b'H'
                           or letter == b'G']
        return terminal_states

    def reset(self):
        raw_obs = super().reset()
        return self._convert_raw_obs_to_x_y_coordinates(raw_obs)

    def _convert_raw_obs_to_x_y_coordinates(self, raw_obs):
        # raw obs is number indicating idx into flattened grid, where 0 is top
        # left, and flattening is done left to right, top to bottom.
        # x is the column coordinate, y is the row coordinate, both starting
        # from 0.
        assert len(raw_obs) == 1
        obs_val = raw_obs[0]
        x = obs_val % self._GRID_SIZE
        y = math.floor(obs_val / self._GRID_SIZE)
        assert (y * self._GRID_SIZE + x) == obs_val
        return np.asarray([x, y])

    def step(self, action):
        raw_response = super().step(action)
        raw_obs = raw_response.obs
        converted_obs = self._convert_raw_obs_to_x_y_coordinates(raw_obs)
        return EnvironmentResponse(obs=converted_obs,
                                   reward=raw_response.reward,
                                   is_terminal=raw_response.is_terminal)


class FrozenLake4x4(FrozenLakeABC):
    _GYM_ENV_NAME = "FrozenLake-v0"
    _GRID_SIZE = 4

    @property
    def min_perf(self):
        raise NotImplementedError

    @property
    def max_perf(self):
        raise NotImplementedError

    def assess_perf(self, policy):
        raise NotImplementedError

    def assess_perf_and_get_trajs(self, policy):
        raise NotImplementedError


class FrozenLake8x8(FrozenLakeABC):
    _GYM_ENV_NAME = "FrozenLake8x8-v0"
    _GRID_SIZE = 8

    @property
    def min_perf(self):
        raise NotImplementedError

    @property
    def max_perf(self):
        raise NotImplementedError

    def assess_perf(self, policy):
        raise NotImplementedError

    def assess_perf_and_get_trajs(self, policy):
        raise NotImplementedError


class FrozenLake16x16(FrozenLakeABC):
    _GYM_ENV_NAME = "FrozenLake16x16-v0"
    _GRID_SIZE = 16

    @property
    def min_perf(self):
        raise NotImplementedError

    @property
    def max_perf(self):
        raise NotImplementedError

    def assess_perf(self, policy):
        raise NotImplementedError

    def assess_perf_and_get_trajs(self, policy):
        raise NotImplementedError
