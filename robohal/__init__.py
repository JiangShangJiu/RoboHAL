"""RoboHAL: synchronous robot hardware interfaces.

``robohal.api`` holds the backend-independent protocol and dataclasses and
imports without MuJoCo. ``MujocoHardware`` (the simulator) imports MuJoCo only
when actually requested, so asset discovery and CLI help stay lightweight.
"""

__all__ = [
    "MujocoHardware",
    "RealHardware",
    "RobotBackend",
    "RobotHardware",
    "JointConfig",
    "JointInfo",
    "JointState",
    "HardwareState",
]

# Names available without importing MuJoCo or a graphics backend.
_BACKEND_FREE = frozenset({
    "RealHardware", "RobotBackend", "RobotHardware",
    "JointConfig", "JointInfo", "JointState", "HardwareState",
})


def __getattr__(name):
    if name == "MujocoHardware":
        from . import hardware
        value = hardware.MujocoHardware
    elif name in _BACKEND_FREE:
        from . import api
        value = getattr(api, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
