class ObsSpaceBuilder:
    """Convenience class to make syntax of building obs space nicer.

    An obs space is just a tuple of dimensions."""
    def __init__(self):
        self._dims = []

    def add_dim(self, dimension):
        self._dims.append(dimension)

    def create_space(self):
        return tuple(self._dims)
