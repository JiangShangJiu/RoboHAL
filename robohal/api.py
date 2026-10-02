"""Backend-independent robot hardware interfaces.

Importing this module never requires MuJoCo or a graphics backend. It defines:

- the immutable command/state dataclasses shared by every backend;
- ``RobotHardware``: the synchronous, joint-named interface a controller
  program should depend on, so the same code drives a simulation or a real
  robot;
- ``RobotBackend``: the small vendor/transport seam a physical robot has to
  implement (bus writes, encoder reads, enable/disable);
- ``RealHardware``: a ``RobotHardware`` built on a ``RobotBackend`` that adds
  unit/limit validation, command latching, mode arbitration and a latched drive
  stop.

The simulated implementation, ``robohal.hardware.MujocoHardware``, satisfies
``RobotHardware`` structurally; ``tests/test_api.py`` enforces conformance.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class JointConfig:
    """Drive gains and limits for one powered joint, in SI units."""

    kp: float = 100.0
    kd: float = 10.0
    effort_limit: float = 100.0
    velocity_limit: float = 2.0

    def __post_init__(self) -> None:
        for name in ("kp", "kd", "effort_limit", "velocity_limit"):
            value = getattr(self, name)
            if not np.isscalar(value) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite scalar")
            if value < 0 or (name.endswith("limit") and value == 0):
                raise ValueError(f"{name} must be {'positive' if name.endswith('limit') else 'nonnegative'}")


@dataclass(frozen=True)
class JointInfo:
    """Static description of one scalar joint.

    ``lower``/``upper`` are position limits in rad/m, ``effort_limit`` is Nm/N
    and ``velocity_limit`` is rad/s or m/s. A ``None`` limit means the joint
    publishes no bound and the corresponding command mode is rejected.
    """

    name: str
    kind: str
    lower: float | None
    upper: float | None
    effort_limit: float | None
    velocity_limit: float | None
    controllable: bool


@dataclass(frozen=True)
class JointState:
    """One joint's feedback sample. ``effort`` is the last applied command."""

    position: float
    velocity: float
    effort: float
    mode: str


@dataclass(frozen=True)
class HardwareState:
    """A consistent snapshot: one timestep, every joint, plus the stop latch."""

    time: float
    joints: dict[str, JointState]
    stopped: bool


@runtime_checkable
class RobotHardware(Protocol):
    """Joint-named, SI-unit, synchronous control interface.

    Commands are latched: ``set_*`` records a target and every subsequent
    ``step`` keeps applying it until replaced. ``step`` advances the machine
    and returns the resulting feedback; it is the single unit of time for both
    a simulation step and one real control cycle.
    """

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Every scalar joint, commanded or not."""
        ...

    @property
    def controlled_joints(self) -> tuple[str, ...]:
        """Joints that currently accept commands."""
        ...

    @property
    def joint_info(self) -> dict[str, JointInfo]:
        """Per-joint metadata keyed by joint name."""
        ...

    @property
    def timestep(self) -> float:
        """Nominal control period in seconds."""
        ...

    def set_positions(self, values: Mapping[str, float]) -> None:
        """Latch position targets in rad or m."""
        ...

    def set_velocities(self, values: Mapping[str, float]) -> None:
        """Latch velocity targets in rad/s or m/s."""
        ...

    def set_efforts(self, values: Mapping[str, float]) -> None:
        """Latch joint torques (Nm) or forces (N)."""
        ...

    def step(self, steps: int = 1) -> HardwareState:
        """Advance ``steps`` control cycles and return the new state."""
        ...

    def read(self) -> HardwareState:
        """Return the latest snapshot without advancing time."""
        ...

    def reset(self, positions: Mapping[str, float] | None = None) -> HardwareState:
        """Return to a known reference state, discarding latched commands."""
        ...

    def stop(self) -> None:
        """Latch drive power off; commands are refused until ``clear_stop``."""
        ...

    def clear_stop(self) -> None:
        """Release the stop latch, holding current positions."""
        ...

    def close(self) -> None:
        """Release the device. Must be safe to call more than once."""
        ...

    def __enter__(self) -> RobotHardware:
        ...

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        ...


@runtime_checkable
class RobotBackend(Protocol):
    """Vendor/transport seam for one physical robot.

    Implementations speak the actual bus (EtherCAT, CAN, serial, vendor SDK).
    ``write`` and ``read`` are one transaction each and must not block longer
    than the control period without surfacing a timeout. All quantities are SI
    and all joints are named; address mapping is the backend's business.
    """

    @property
    def control_period(self) -> float:
        """Target cycle time in seconds."""
        ...

    def describe(self) -> Sequence[JointInfo]:
        """Static joint inventory; called once when the device is opened."""
        ...

    def enable(self) -> None:
        """Power the drives so they accept commands."""
        ...

    def disable(self) -> None:
        """Remove drive power. Must be idempotent and safe from any state."""
        ...

    def write(self, commands: Mapping[str, tuple[str, float]]) -> None:
        """Send latched targets as ``{joint: (mode, value)}`` in SI units."""
        ...

    def read(self) -> tuple[float, Mapping[str, JointState]]:
        """Return ``(time_seconds, {joint: JointState})`` for every joint."""
        ...

    def close(self) -> None:
        """Release the bus. Must be safe to call more than once."""
        ...


class RealHardware:
    """``RobotHardware`` on top of a ``RobotBackend``.

    This class owns everything that is protocol-independent: validating SI
    commands against ``JointInfo``, latching targets, arbitrating one mode per
    joint, holding position after a stop, and refusing commands while the stop
    latch is set. A vendor integration only has to provide a ``RobotBackend``.

    Opening the device enables the drives and latches a hold at the current
    pose, mirroring ``MujocoHardware`` where drives are active from
    construction. Call ``stop()`` or ``close()`` to remove drive power.

    Unlike the simulator, a real device cannot be teleported to a commanded
    pose, so ``reset(positions=...)`` is refused; move there with
    ``set_positions`` and ``step`` instead.
    """

    def __init__(self, backend: RobotBackend) -> None:
        self._backend = backend
        period = backend.control_period
        if not np.isscalar(period) or not np.isfinite(period) or period <= 0:
            raise ValueError("control_period must be finite and positive")
        self._timestep = float(period)

        infos = tuple(backend.describe())
        if not infos:
            raise ValueError("Backend describes no joints")
        self._info: dict[str, JointInfo] = {}
        for info in infos:
            if info.name in self._info:
                raise ValueError(f"Backend describes duplicate joint: {info.name}")
            self._info[info.name] = info

        self._closed = False
        self._stopped = False
        self._commands: dict[str, tuple[str, float]] = {}
        self._state: dict[str, JointState] = {}
        self._time = 0.0
        self._store(*backend.read())
        self._hold()
        # Parity with the simulator: the drives are live and holding the pose
        # observed at open, so a controller can command immediately.
        backend.enable()
        if self._commands:
            backend.write(dict(self._commands))

    # -- introspection -----------------------------------------------------

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(self._info)

    @property
    def controlled_joints(self) -> tuple[str, ...]:
        return tuple(name for name, info in self._info.items() if info.controllable)

    @property
    def joint_info(self) -> dict[str, JointInfo]:
        return self._info.copy()

    @property
    def timestep(self) -> float:
        return self._timestep

    # -- commands ----------------------------------------------------------

    def set_positions(self, values: Mapping[str, float]) -> None:
        """Latch position targets in rad/m. Unspecified joints are unchanged."""
        self._set(values, "position")

    def set_velocities(self, values: Mapping[str, float]) -> None:
        """Latch velocity targets in rad/s or m/s."""
        self._set(values, "velocity")

    def set_efforts(self, values: Mapping[str, float]) -> None:
        """Latch joint torques (Nm) or forces (N)."""
        self._set(values, "effort")

    # -- clock -------------------------------------------------------------

    def step(self, steps: int = 1) -> HardwareState:
        """Run ``steps`` write/read cycles and return the resulting state."""
        self._assert_open()
        if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        for _ in range(steps):
            if not self._stopped:
                self._backend.write(dict(self._commands))
            self._store(*self._backend.read())
        return self.read()

    def read(self) -> HardwareState:
        """Return the latest snapshot without touching the bus."""
        self._assert_open()
        joints = {}
        for name, state in self._state.items():
            commanded = name in self._commands and self._info[name].controllable
            joints[name] = JointState(
                position=state.position,
                velocity=state.velocity,
                effort=state.effort,
                mode=self._commands[name][0] if commanded else "disabled",
            )
        return HardwareState(time=self._time, joints=joints, stopped=self._stopped)

    def reset(self, positions: Mapping[str, float] | None = None) -> HardwareState:
        """Discard latched commands and hold current positions.

        ``positions`` is refused: a real device cannot jump to a commanded
        pose. Use ``set_positions`` followed by ``step`` to move there.
        """
        self._assert_open()
        if positions is not None:
            raise ValueError(
                "RealHardware cannot jump to positions; command a move with "
                "set_positions and step"
            )
        self._hold()
        return self.read()

    # -- safety ------------------------------------------------------------

    def stop(self) -> None:
        """Latch drive power off. Idempotent; physics/bus reads continue."""
        self._assert_open()
        self._backend.disable()
        self._stopped = True
        self._hold()

    def clear_stop(self) -> None:
        """Release the stop latch and re-enable drives holding current poses."""
        self._assert_open()
        self._backend.enable()
        self._stopped = False
        self._hold()

    def close(self) -> None:
        if self._closed:
            return
        self._backend.disable()
        self._stopped = True
        self._backend.close()
        self._closed = True

    def __enter__(self) -> RealHardware:
        self._assert_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    # -- internals ---------------------------------------------------------

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("Hardware is closed")

    def _store(self, time: float, states: Mapping[str, JointState]) -> None:
        missing = [name for name in self._info if name not in states]
        if missing:
            raise ValueError(f"Backend omitted joints: {missing}")
        value = float(time)
        if not np.isfinite(value):
            raise ValueError("Backend returned a non-finite time")
        self._time = value
        self._state = {name: states[name] for name in self._info}

    def _hold(self) -> None:
        self._commands = {}
        for name, info in self._info.items():
            if not info.controllable:
                continue
            position = self._state[name].position
            if info.lower is not None:
                position = min(max(position, info.lower), info.upper)
            self._commands[name] = ("disabled" if self._stopped else "position", position)

    def _validate(self, values: Mapping[str, float], mode: str) -> dict[str, float]:
        if not isinstance(values, Mapping):
            raise ValueError("Commands must map joint names to scalar values")
        checked = {}
        for name, value in values.items():
            info = self._info.get(name)
            if info is None:
                raise ValueError(f"Unknown joint: {name}")
            if not info.controllable:
                raise ValueError(f"Passive joint: {name}")
            try:
                if isinstance(value, (str, bytes, bool)) or not np.isscalar(value):
                    raise ValueError()
                value = float(value)
                if not np.isfinite(value):
                    raise ValueError()
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Non-finite or non-scalar command for {name}") from exc
            if mode == "position":
                if info.lower is not None and not info.lower <= value <= info.upper:
                    raise ValueError(f"{name}: position {value} outside [{info.lower}, {info.upper}]")
            else:
                limit = getattr(info, f"{mode}_limit")
                if limit is None:
                    raise ValueError(f"{name}: no {mode} limit published; refusing the command")
                if abs(value) > limit:
                    raise ValueError(f"{name}: {mode} {value} exceeds +/-{limit}")
            checked[name] = value
        return checked

    def _set(self, values: Mapping[str, float], mode: str) -> None:
        self._assert_open()
        if self._stopped:
            raise RuntimeError("Emergency stop is latched; call clear_stop() first")
        checked = self._validate(values, mode)
        self._commands.update({name: (mode, value) for name, value in checked.items()})


__all__ = [
    "HardwareState",
    "JointConfig",
    "JointInfo",
    "JointState",
    "RealHardware",
    "RobotBackend",
    "RobotHardware",
]
