class ObsSpaceBase:
    def __init__(self, dims):
        self._dims = tuple(dims)

    def __iter__(self):
        return iter(self._dims)


class IntegerObsSpace(ObsSpaceBase):
    pass


class RealObsSpace(ObsSpaceBase):
    pass


class ObsSpaceBuilder:
    """Convenience class to make syntax of building obs space nicer."""
    def __init__(self):
        self._dims = []

    def add_dim(self, dim):
        self._dims.append(dim)

    def create_integer_space(self):
        return IntegerObsSpace(self._dims)

    def create_real_space(self):
        return RealObsSpace(self._dims)
