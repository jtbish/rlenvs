import numpy as np

from .dimension import RealDimension
from .environment import EnvironmentABC, EnvironmentResponse
from .obs_space import RealObsSpace, RealObsSpaceBuilder

_OBS_DIM_LOWER = 0.0
_OBS_DIM_UPPER = 1.0


class NormaliseWrapper(EnvironmentABC):
    """Wrapper over EnvironmentABC derived class that uses a real obs space to
    min-max normalise all observations between [0, 1]."""
    def __init__(self, wrapped):
        assert isinstance(wrapped.obs_space, RealObsSpace)
        self._wrapped = wrapped
        self._unit_hypercube_obs_space = \
            self._gen_unit_hypercube_obs_space(self._wrapped.obs_space)

    def _gen_unit_hypercube_obs_space(self, wrapped_obs_space):
        builder = RealObsSpaceBuilder()
        for dim in wrapped_obs_space:
            builder.add_dim(
                RealDimension(_OBS_DIM_LOWER, _OBS_DIM_UPPER, dim.name))
        return builder.create_space()

    def _normalise_wrapped_obs(self, wrapped_obs):
        normalised = []
        for (obs_compt, dim) in zip(wrapped_obs, self._wrapped.obs_space):
            new_val = (obs_compt - dim.lower) / dim.span
            assert _OBS_DIM_LOWER <= new_val <= _OBS_DIM_UPPER
            normalised.append(new_val)
        return np.array(normalised)

    @property
    def perf_lower_bound(self):
        return self._wrapped.perf_lower_bound

    def _sample_initial_obs(self):
        # don't need to normalise initial obss since only used internally
        return self._wrapped._sample_initial_obs()

    @property
    def time_limit(self):
        return self._wrapped.time_limit

    @property
    def obs_space(self):
        """Use non-wrapped obs space!"""
        return self._unit_hypercube_obs_space

    @property
    def action_space(self):
        return self._wrapped.action_space

    def reset(self):
        wrapped_obs = self._wrapped.reset()
        return self._normalise_wrapped_obs(wrapped_obs)

    def step(self, action):
        wrapped_response = self._wrapped.step(action)
        return EnvironmentResponse(obs=self._normalise_wrapped_obs(
            wrapped_response.obs),
                                   reward=wrapped_response.reward,
                                   is_terminal=wrapped_response.is_terminal,
                                   info=wrapped_response.info)

    def is_terminal(self):
        return self._wrapped.is_terminal()

    def render(self):
        self._wrapped.render()

    def reseed_iod_rng(self, new_seed):
        self._wrapped.reseed_iod_rng(new_seed)
