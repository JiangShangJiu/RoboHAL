"""Synchronous MuJoCo hardware with named, ideal joint drives.

Commands are SI quantities. Drives apply generalized effort through
``qfrc_applied``; native MJCF actuators are disabled to avoid double actuation.
This is a software servo, not an electrical motor or vendor firmware model.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Mapping

import mujoco
import numpy as np


@dataclass(frozen=True)
class JointConfig:
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
    name: str
    kind: str
    lower: float | None
    upper: float | None
    effort_limit: float | None
    velocity_limit: float | None
    controllable: bool


@dataclass(frozen=True)
class JointState:
    position: float
    velocity: float
    effort: float
    mode: str


@dataclass(frozen=True)
class HardwareState:
    time: float
    joints: dict[str, JointState]
    stopped: bool


class MujocoHardware:
    """One hardware instance, one simulation clock, no background thread.

    By default, every named hinge/slide joint receives an ideal drive. Pass
    ``joints={name: JointConfig(...)}`` to select actual powered joints and
    leave all others passive. Ball/free joints are never implicitly driven.
    ``effort`` feedback is the drive's last applied torque/force, not a contact
    wrench or a physical torque sensor. The model is copied on construction.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        joints: Mapping[str, JointConfig] | None = None,
        timestep: float | None = None,
        home_keyframe: str | None = "home",
        gravity_compensation: bool = False,
    ) -> None:
        self.model = copy.copy(model)
        if timestep is not None:
            if not np.isscalar(timestep) or not np.isfinite(timestep) or timestep <= 0:
                raise ValueError("timestep must be finite and positive")
            self.model.opt.timestep = float(timestep)
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        self.data = mujoco.MjData(self.model)
        self._closed = False
        self._stopped = False
        self._renderer = None
        self._render_size = None
        self._compensate = bool(gravity_compensation)
        self._joint_ids: dict[str, int] = {}
        self._qadr: dict[str, int] = {}
        self._vadr: dict[str, int] = {}
        for jid in range(self.model.njnt):
            if self.model.jnt_type[jid] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if name:
                self._joint_ids[name] = jid
                self._qadr[name] = int(self.model.jnt_qposadr[jid])
                self._vadr[name] = int(self.model.jnt_dofadr[jid])

        if joints is None:
            self._configs = {name: self._default_config(jid) for name, jid in self._joint_ids.items()}
        else:
            self._configs = dict(joints)
            for name, config in self._configs.items():
                if name not in self._joint_ids:
                    raise ValueError(f"Unknown or non-scalar joint: {name}")
                if not isinstance(config, JointConfig):
                    raise ValueError(f"Expected JointConfig for {name}")

        self._info = {}
        for name, jid in self._joint_ids.items():
            limited = bool(self.model.jnt_limited[jid])
            config = self._configs.get(name)
            self._info[name] = JointInfo(
                name=name,
                kind="hinge" if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE else "slide",
                lower=float(self.model.jnt_range[jid, 0]) if limited else None,
                upper=float(self.model.jnt_range[jid, 1]) if limited else None,
                effort_limit=config.effort_limit if config else None,
                velocity_limit=config.velocity_limit if config else None,
                controllable=config is not None,
            )
        self._home_id = -1
        if home_keyframe is not None:
            self._home_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, home_keyframe)
            if self._home_id < 0 and home_keyframe != "home":
                raise ValueError(f"Unknown home keyframe: {home_keyframe}")
        self._commands: dict[str, tuple[str, float]] = {}
        self.reset()

    @classmethod
    def from_robot(cls, robot: str = "panda", scene: str = "empty.xml", **kwargs) -> MujocoHardware:
        from .models import load_robot

        return cls(load_robot(robot, scene), **kwargs)

    @classmethod
    def from_xml(cls, path: str | Path, **kwargs) -> MujocoHardware:
        from .models import load_mjcf

        return cls(load_mjcf(path), **kwargs)

    def _default_config(self, jid: int) -> JointConfig:
        # Reuse symmetric direct-drive force bounds when available. Software
        # gains and speed limits remain simulation defaults, not vendor specs.
        limits = []
        if self.model.jnt_actfrclimited[jid]:
            low, high = self.model.jnt_actfrcrange[jid]
            if low < 0 < high:
                limits.append(min(-low, high))
        for aid in range(self.model.nu):
            if (self.model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT
                    and self.model.actuator_trnid[aid, 0] == jid
                    and self.model.actuator_forcelimited[aid]):
                low, high = self.model.actuator_forcerange[aid]
                gear = abs(float(self.model.actuator_gear[aid, 0]))
                if low < 0 < high and gear > 0:
                    limits.append(min(-low, high) * gear)
        effort = float(min(limits)) if limits else 100.0
        if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE:
            return JointConfig(kp=500.0, kd=20.0, effort_limit=effort, velocity_limit=0.5)
        return JointConfig(effort_limit=effort)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(self._joint_ids)

    @property
    def controlled_joints(self) -> tuple[str, ...]:
        return tuple(self._configs)

    @property
    def joint_info(self) -> dict[str, JointInfo]:
        return self._info.copy()

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("Hardware is closed")

    def _validate(self, values: Mapping[str, float], mode: str) -> dict[str, float]:
        if not isinstance(values, Mapping):
            raise ValueError("Commands must map joint names to scalar values")
        checked = {}
        for name, value in values.items():
            if name not in self._configs:
                raise ValueError(f"Unknown or passive joint: {name}")
            try:
                if isinstance(value, (str, bytes, bool)) or not np.isscalar(value):
                    raise ValueError()
                value = float(value)
                if not np.isfinite(value):
                    raise ValueError()
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Non-finite or non-scalar command for {name}") from exc
            info = self._info[name]
            if mode == "position":
                if info.lower is not None and not info.lower <= value <= info.upper:
                    raise ValueError(f"{name}: position {value} outside [{info.lower}, {info.upper}]")
            else:
                limit = getattr(self._configs[name], f"{mode}_limit")
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

    def set_positions(self, values: Mapping[str, float]) -> None:
        """Latch targets in rad/m. Unspecified joints retain their commands."""
        self._set(values, "position")

    def set_velocities(self, values: Mapping[str, float]) -> None:
        """Latch velocity targets in rad/s or m/s."""
        self._set(values, "velocity")

    def set_efforts(self, values: Mapping[str, float]) -> None:
        """Latch joint torques (Nm) or forces (N); no bias compensation."""
        self._set(values, "effort")

    def _hold(self) -> None:
        self._commands = {}
        for name in self._configs:
            info = self._info[name]
            position = float(self.data.qpos[self._qadr[name]])
            if info.lower is not None:
                position = float(np.clip(position, info.lower, info.upper))
            self._commands[name] = ("disabled" if self._stopped else "position", position)

    def reset(self, positions: Mapping[str, float] | None = None) -> HardwareState:
        """Reset time and dynamics, preserving a latched emergency stop.

        Uses the home keyframe when present, otherwise qpos0. Scalar joint
        initial positions outside their limits are projected into the range.
        Supplied positions are checked atomically before changing any state.
        """
        self._assert_open()
        checked = self._validate(positions, "position") if positions is not None else {}
        if self._home_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        self.data.time = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.act[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        for name, info in self._info.items():
            if info.lower is not None:
                adr = self._qadr[name]
                self.data.qpos[adr] = np.clip(self.data.qpos[adr], info.lower, info.upper)
        for name, value in checked.items():
            self.data.qpos[self._qadr[name]] = value
        self._hold()
        mujoco.mj_forward(self.model, self.data)
        return self.read()

    def _apply_drives(self) -> None:
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        if self._stopped:
            return
        for name, config in self._configs.items():
            mode, target = self._commands[name]
            q = float(self.data.qpos[self._qadr[name]])
            vadr = self._vadr[name]
            v = float(self.data.qvel[vadr])
            if mode == "position":
                # Saturate the PD position term via a desired velocity. The
                # gain ratio preserves kp*error - kd*v below the speed bound.
                if config.kd > 0:
                    desired_v = np.clip(config.kp * (target - q) / config.kd,
                                        -config.velocity_limit, config.velocity_limit)
                    effort = config.kd * (desired_v - v)
                else:
                    effort = config.kp * (target - q)
            elif mode == "velocity":
                effort = config.kd * (target - v)
            elif mode == "effort":
                effort = target
            else:
                continue
            if self._compensate and mode in ("position", "velocity"):
                effort += self.data.qfrc_bias[vadr]
            effort = float(np.clip(effort, -config.effort_limit, config.effort_limit))
            info = self._info[name]
            if info.lower is not None:
                if (q <= info.lower and effort < 0) or (q >= info.upper and effort > 0):
                    effort = 0.0
            self.data.qfrc_applied[vadr] = effort

    def step(self, steps: int = 1) -> HardwareState:
        """Advance exactly ``steps * timestep`` seconds without sleeping."""
        self._assert_open()
        if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        for _ in range(steps):
            before = float(self.data.time)
            self._apply_drives()
            mujoco.mj_step(self.model, self.data)
            if (not np.all(np.isfinite(self.data.qpos))
                    or not np.all(np.isfinite(self.data.qvel))
                    or not np.isclose(self.data.time, before + self.timestep, rtol=0, atol=1e-10)):
                self.stop()
                raise RuntimeError("MuJoCo simulation became unstable; drives stopped")
            # mj_step integrates qpos/qvel after its forward pass. Recompute
            # derived poses/sensors so feedback matches the returned timestamp.
            mujoco.mj_forward(self.model, self.data)
        return self.read()

    def read(self) -> HardwareState:
        self._assert_open()
        return HardwareState(
            time=float(self.data.time),
            joints={name: JointState(
                position=float(self.data.qpos[self._qadr[name]]),
                velocity=float(self.data.qvel[self._vadr[name]]),
                effort=float(self.data.qfrc_applied[self._vadr[name]]) if name in self._configs else 0.0,
                mode=self._commands[name][0] if name in self._commands else "disabled",
            ) for name in self._joint_ids},
            stopped=self._stopped,
        )

    def read_sensors(self) -> dict[str, np.ndarray]:
        """Return copies of named MJCF sensors (IMU, force, touch, etc.)."""
        self._assert_open()
        sensors = {}
        for sid in range(self.model.nsensor):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sid)
            if name:
                adr = int(self.model.sensor_adr[sid])
                dim = int(self.model.sensor_dim[sid])
                sensors[name] = self.data.sensordata[adr:adr + dim].copy()
        return sensors

    def body_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """World position (m) and quaternion (w, x, y, z) of a body."""
        self._assert_open()
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"Unknown body: {name}")
        return self.data.xpos[bid].copy(), self.data.xquat[bid].copy()

    def render_camera(self, camera: str | int = -1, *, width: int = 640, height: int = 480) -> np.ndarray:
        """RGB uint8 image. A working OpenGL/EGL backend is required."""
        self._assert_open()
        if self._render_size != (width, height):
            if self._renderer is not None:
                self._renderer.close()
                self._renderer = None
            self._renderer = mujoco.Renderer(self.model, width=width, height=height)
            self._render_size = (width, height)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()

    def stop(self) -> None:
        """Latch drive power off. Physics continues; this is not a brake."""
        self._assert_open()
        self._stopped = True
        self.data.qfrc_applied[:] = 0.0
        self.data.ctrl[:] = 0.0
        self._hold()
        mujoco.mj_forward(self.model, self.data)

    def clear_stop(self) -> None:
        """Re-enable drives holding current positions; discard old commands."""
        self._assert_open()
        self._stopped = False
        self._hold()

    def close(self) -> None:
        if not self._closed:
            self.stop()
            if self._renderer is not None:
                self._renderer.close()
            self._closed = True

    def __enter__(self) -> MujocoHardware:
        self._assert_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
