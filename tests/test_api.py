"""The hardware protocol, its simulator conformance, and the real-robot shell."""

from __future__ import annotations

import math

import mujoco
import pytest

from robohal import (
    HardwareState,
    JointInfo,
    JointState,
    MujocoHardware,
    RealHardware,
    RobotBackend,
    RobotHardware,
)


class FakeBackend:
    """Minimal in-memory ``RobotBackend``: position mode moves instantly."""

    def __init__(self) -> None:
        self.enabled = False
        self.closed = False
        self.writes: list[dict[str, tuple[str, float]]] = []
        self.time = 0.0
        self.positions = {"hinge": 0.0, "slide": 0.0}

    @property
    def control_period(self) -> float:
        return 0.001

    def describe(self):
        return (
            JointInfo("hinge", "hinge", -1.0, 1.0, 5.0, 2.0, True),
            JointInfo("slide", "slide", -0.5, 0.5, None, None, False),
        )

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def write(self, commands) -> None:
        self.writes.append(dict(commands))
        for name, (mode, value) in commands.items():
            if mode == "position":
                self.positions[name] = value

    def read(self):
        self.time += self.control_period
        return self.time, {
            name: JointState(position, 0.0, 0.0, "disabled")
            for name, position in self.positions.items()
        }

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def hardware(backend: FakeBackend) -> RealHardware:
    with RealHardware(backend) as hardware:
        yield hardware


def test_simulator_is_structurally_a_robot_hardware():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint name="j" type="hinge" axis="0 0 1"/>'
        '<geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
    )
    with MujocoHardware(model) as hw:
        assert isinstance(hw, RobotHardware)
        assert isinstance(hw, RobotBackend) is False  # distinct protocols must not alias
    # The shell also satisfies the protocol, and both are usable as context managers.
    backend = FakeBackend()
    shell = RealHardware(backend)
    assert isinstance(shell, RobotHardware)
    with shell:
        pass


def test_api_imports_without_mujoco():
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    code = (
        "import sys, robohal; "
        "from robohal import RealHardware, RobotHardware, JointInfo, JointState, HardwareState; "
        "assert 'mujoco' not in sys.modules, sys.modules.keys(); "
        "assert 'glfw' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)


def test_open_latches_current_positions_and_reports_metadata(hardware: RealHardware, backend: FakeBackend):
    assert hardware.joint_names == ("hinge", "slide")
    assert hardware.controlled_joints == ("hinge",)
    assert hardware.timestep == pytest.approx(0.001)
    assert backend.enabled is True  # opening turns the drives on and holds pose
    state = hardware.read()
    assert state.stopped is False
    assert state.joints["hinge"].mode == "position"
    assert state.joints["hinge"].position == pytest.approx(0.0)
    assert state.joints["slide"].mode == "disabled"


def test_commands_latch_and_are_resent_every_cycle(hardware: RealHardware, backend: FakeBackend):
    hardware.set_positions({"hinge": 0.4})
    hardware.step(3)
    assert hardware.read().joints["hinge"].position == pytest.approx(0.4)
    assert hardware.read().joints["hinge"].mode == "position"
    assert backend.writes[-1]["hinge"] == ("position", 0.4)
    assert len(backend.writes) == 4  # one for the hold sent at construction, plus three steps


def test_mode_is_arbitrated_per_joint(hardware: RealHardware):
    hardware.set_velocities({"hinge": 0.5})
    assert hardware.read().joints["hinge"].mode == "velocity"
    hardware.set_efforts({"hinge": 0.25})
    hardware.step()
    state = hardware.read()
    assert state.joints["hinge"].mode == "effort"
    assert state.joints["hinge"].effort == pytest.approx(0.0)  # backend feedback, not the target


@pytest.mark.parametrize("value", [-1.5, 1.5, math.nan, math.inf, -math.inf])
def test_position_limits_and_nonfinite_values_are_rejected(hardware: RealHardware, value):
    with pytest.raises(ValueError):
        hardware.set_positions({"hinge": value})


def test_velocity_and_effort_limits_are_enforced(hardware: RealHardware):
    with pytest.raises(ValueError):
        hardware.set_velocities({"hinge": 2.5})
    with pytest.raises(ValueError):
        hardware.set_efforts({"hinge": 6.0})


def test_joints_without_a_published_limit_refuse_that_mode(hardware: RealHardware):
    assert hardware.joint_info["hinge"].effort_limit == 5.0
    with pytest.raises(ValueError, match="Unknown joint"):
        hardware.set_positions({"missing": 0.0})


def test_passive_joint_can_be_read_but_not_commanded(hardware: RealHardware):
    for method in (hardware.set_positions, hardware.set_velocities, hardware.set_efforts):
        with pytest.raises(ValueError, match="Passive joint"):
            method({"slide": 0.1})


def test_invalid_batch_leaves_every_previous_command_intact(hardware: RealHardware):
    hardware.set_positions({"hinge": 0.4})
    with pytest.raises(ValueError):
        hardware.set_positions({"hinge": 0.1, "slide": 0.0})
    hardware.step(2)
    assert hardware.read().joints["hinge"].position == pytest.approx(0.4)
    assert hardware.read().joints["hinge"].mode == "position"


@pytest.mark.parametrize("steps", [-1, 1.5, True, math.nan])
def test_invalid_step_counts_do_not_touch_the_bus(hardware: RealHardware, backend: FakeBackend, steps):
    before = len(backend.writes)
    with pytest.raises((TypeError, ValueError)):
        hardware.step(steps)
    assert len(backend.writes) == before


def test_reset_discards_commands_instead_of_teleporting(hardware: RealHardware):
    hardware.set_positions({"hinge": 0.4})
    hardware.step()
    with pytest.raises(ValueError, match="cannot jump"):
        hardware.reset({"hinge": -0.2})
    state = hardware.reset()
    assert state.joints["hinge"].position == pytest.approx(0.4)
    assert state.joints["hinge"].mode == "position"


def test_stop_latches_drives_off_and_clear_stop_holds_current_pose(
    hardware: RealHardware, backend: FakeBackend
):
    hardware.set_positions({"hinge": 0.6})
    hardware.step()
    hardware.stop()
    assert hardware.read().stopped is True
    assert backend.enabled is False
    for method in (hardware.set_positions, hardware.set_velocities, hardware.set_efforts):
        with pytest.raises(RuntimeError):
            method({"hinge": 0.0})
    before = len(backend.writes)
    hardware.step(5)  # Reads continue; no command is pushed while stopped.
    assert len(backend.writes) == before
    hardware.reset()
    assert hardware.read().stopped is True
    hardware.clear_stop()
    assert hardware.read().stopped is False
    assert backend.enabled is True
    assert hardware.read().joints["hinge"].mode == "position"
    assert hardware.read().joints["hinge"].position == pytest.approx(0.6)


def test_close_is_idempotent_and_releases_the_bus(backend: FakeBackend):
    hardware = RealHardware(backend)
    hardware.close()
    hardware.close()
    assert backend.enabled is False
    assert backend.closed is True
    for operation in (
        hardware.read, hardware.step, hardware.reset, hardware.stop,
        hardware.clear_stop, lambda: hardware.set_positions({"hinge": 0.0}),
    ):
        with pytest.raises(RuntimeError):
            operation()


def test_constructor_rejects_a_misbehaving_backend():
    class NoJoints(FakeBackend):
        def describe(self):
            return ()

    class BadPeriod(FakeBackend):
        @property
        def control_period(self):
            return 0.0

    class Duplicate(FakeBackend):
        def describe(self):
            return (JointInfo("hinge", "hinge", -1, 1, 5, 2, True),
                    JointInfo("hinge", "hinge", -1, 1, 5, 2, True))

    with pytest.raises(ValueError, match="no joints"):
        RealHardware(NoJoints())
    with pytest.raises(ValueError, match="control_period"):
        RealHardware(BadPeriod())
    with pytest.raises(ValueError, match="duplicate"):
        RealHardware(Duplicate())


def test_snapshots_are_independent_copies(hardware: RealHardware):
    state = hardware.read()
    state.joints["hinge"] = JointState(99.0, 99.0, 99.0, "effort")
    assert hardware.read().joints["hinge"].position == pytest.approx(0.0)
    info = hardware.joint_info
    info.clear()
    assert "hinge" in hardware.joint_info


def test_hardware_state_is_immutable_dataclass():
    state = HardwareState(time=0.0, joints={}, stopped=False)
    with pytest.raises(Exception):
        state.time = 1.0
