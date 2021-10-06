import abc


class DimensionABC(metaclass=abc.ABCMeta):
    def __init__(self, lower, upper, name):
        assert lower <= upper
        self._lower = lower
        self._upper = upper
        self._name = name
        self._span = self._calc_span(self._lower, self._upper)

    @abc.abstractmethod
    def _calc_span(self, lower, upper):
        raise NotImplementedError

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    @property
    def name(self):
        return self._name

    @property
    def span(self):
        return self._span


class IntegerDimension(DimensionABC):
    def _calc_span(self, lower, upper):
        return (upper - lower + 1)


class RealDimension(DimensionABC):
    def _calc_span(self, lower, upper):
        return (upper - lower)
