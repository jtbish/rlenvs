class EnvError(Exception):
    pass


class EndOfEpisodeError(EnvError):
    """Indicates that environment instance has cycled through all steps of
    episode and can no longer be interacted with."""
    pass


class InvalidSpecError(EnvError):
    """Indicates specification of an environment instance was invalid."""
    pass
