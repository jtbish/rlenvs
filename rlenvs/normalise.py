import numpy as np

from .dimension import RealDimension
from .environment import EnvironmentResponse
from .obs_space import RealObsSpace, RealObsSpaceBuilder

_OBS_DIM_LOWER = 0.0
_OBS_DIM_UPPER = 1.0


class NormaliseWrapper:
    """Wrapper over EnvironmentABC derived class that uses a real obs space to
    min-max normalise all observations between [0, 1]."""
    def __init__(self, wrapped):
        assert isinstance(wrapped.obs_space, RealObsSpace)
        self._wrapped = wrapped

        self._dim_lowers_arr = np.asarray(
            [dim.lower for dim in self._wrapped.obs_space])
        self._dim_spans_arr = np.asarray(
            [dim.span for dim in self._wrapped.obs_space])

        self._unit_hypercube_obs_space = \
            self._gen_unit_hypercube_obs_space(self._wrapped.obs_space)

    def _gen_unit_hypercube_obs_space(self, wrapped_obs_space):
        builder = RealObsSpaceBuilder()
        for dim in wrapped_obs_space:
            builder.add_dim(
                RealDimension(_OBS_DIM_LOWER, _OBS_DIM_UPPER, dim.name))
        return builder.create_space()

    def _normalise_wrapped_obs(self, wrapped_obs):
        # numpy arr ops faster than for loop
        return (wrapped_obs - self._dim_lowers_arr) / self._dim_spans_arr

    @property
    def perf_lower_bound(self):
        return self._wrapped.perf_lower_bound

    @property
    def time_limit(self):
        return self._wrapped.time_limit

    @property
    def obs_space(self):
        """Override: use non-wrapped obs space."""
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

    def reseed_wrapped_rng(self, new_seed):
        self._wrapped.reseed_wrapped_rng(new_seed)
