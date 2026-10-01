"""RoboArena: small synchronous hardware interfaces backed by MuJoCo."""

__all__ = ["MujocoHardware", "JointConfig", "JointInfo", "JointState", "HardwareState"]


def __getattr__(name):
    # Asset discovery and CLI help do not need MuJoCo or a graphics backend.
    if name in __all__:
        from . import hardware
        value = getattr(hardware, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
