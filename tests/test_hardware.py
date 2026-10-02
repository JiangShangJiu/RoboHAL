"""Behavioral tests for the hardware API using self-contained MuJoCo models."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from robohal import JointConfig, MujocoHardware


MODEL_XML = """
<mujoco model="hardware_test">
  <compiler angle="radian"/>
  <option gravity="0 0 0" timestep="0.002" integrator="implicitfast"/>
  <default>
    <joint damping="0.05"/>
    <geom contype="0" conaffinity="0"/>
  </default>
  <worldbody>
    <body name="arm" pos="0 0 1">
      <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
      <joint name="hinge" type="hinge" axis="0 0 1" range="-1 1"/>
      <geom type="sphere" size="0.05"/>
    </body>
    <body name="carriage" pos="1 0 1">
      <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
      <joint name="slide" type="slide" axis="1 0 0" range="-0.5 0.5"/>
      <geom type="sphere" size="0.05"/>
    </body>
  </worldbody>
  <actuator>
    <position name="native_drive" joint="hinge" kp="500"/>
  </actuator>
  <sensor>
    <jointpos name="hinge_position" joint="hinge"/>
    <jointvel name="slide_velocity" joint="slide"/>
  </sensor>
  <keyframe><key name="home" qpos="0.1 -0.1"/></keyframe>
</mujoco>
"""

FLOATING_XML = """
<mujoco model="floating_hardware_test">
  <compiler angle="radian"/>
  <option gravity="0 0 0" timestep="0.002"/>
  <worldbody>
    <body name="floating_base" pos="0 0 1">
      <freejoint name="base"/>
      <geom type="sphere" size="0.1" mass="2" contype="0" conaffinity="0"/>
      <body name="link" pos="0 0 0.2">
        <joint name="hinge" type="hinge" axis="0 0 1" range="-1 1"/>
        <geom type="sphere" size="0.1" mass="1" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
  <sensor><jointpos name="hinge_position" joint="hinge"/></sensor>
</mujoco>
"""


@pytest.fixture
def model():
    return mujoco.MjModel.from_xml_string(MODEL_XML)


@pytest.fixture
def hardware(model):
    with MujocoHardware(model) as hardware:
        yield hardware


def test_home_joint_metadata_and_simulation_clock(hardware):
    state = hardware.read()
    assert hardware.joint_names == ("hinge", "slide")
    assert hardware.controlled_joints == ("hinge", "slide")
    assert state.time == 0
    assert state.stopped is False
    assert state.joints["hinge"].position == pytest.approx(0.1)
    assert state.joints["slide"].position == pytest.approx(-0.1)
    assert hardware.joint_info["hinge"].lower == pytest.approx(-1)
    assert hardware.joint_info["slide"].upper == pytest.approx(0.5)
    assert hardware.joint_info["hinge"].controllable is True
    assert hardware.step(0).time == 0
    assert hardware.step(25).time == pytest.approx(0.05)


def test_position_servo_converges_and_recomputes_each_step(hardware):
    hardware.set_positions({"hinge": 0.55, "slide": 0.25})
    state = hardware.step(1000)
    assert state.joints["hinge"].position == pytest.approx(0.55, abs=0.005)
    assert state.joints["slide"].position == pytest.approx(0.25, abs=0.005)
    assert abs(state.joints["hinge"].velocity) < 0.01
    assert abs(state.joints["slide"].velocity) < 0.01
    assert state.joints["hinge"].mode == "position"


def test_velocity_mode_reaches_requested_velocity(hardware):
    hardware.set_velocities({"hinge": 0.6, "slide": 0.2})
    state = hardware.step(250)
    assert state.joints["hinge"].velocity == pytest.approx(0.6, abs=0.03)
    assert state.joints["slide"].velocity == pytest.approx(0.2, abs=0.02)
    assert state.joints["hinge"].mode == "velocity"
    assert state.joints["hinge"].position > 0.3


def test_effort_mode_produces_physical_motion(hardware):
    hardware.set_efforts({"hinge": 0.2, "slide": 2.0})
    state = hardware.step(50)
    assert state.joints["hinge"].velocity > 0.15
    assert state.joints["slide"].velocity > 0.15
    assert state.joints["hinge"].position > 0.105
    assert state.joints["slide"].position > -0.095
    assert state.joints["hinge"].effort == pytest.approx(0.2)
    assert state.joints["slide"].effort == pytest.approx(2)
    assert state.joints["hinge"].mode == "effort"


def test_existing_actuator_cannot_fight_the_hardware_driver(hardware):
    # An enabled native drive would apply approximately 450 Nm at this target.
    hardware.data.ctrl[:] = 1
    hardware.set_efforts({"hinge": 0, "slide": 0})
    state = hardware.step(50)
    np.testing.assert_allclose(hardware.data.qfrc_actuator, 0, atol=1e-12)
    assert state.joints["hinge"].position == pytest.approx(0.1, abs=1e-10)
    assert state.joints["hinge"].velocity == pytest.approx(0, abs=1e-10)


def test_output_force_saturates_without_clipping_position_request(model):
    config = JointConfig(kp=200, kd=0, effort_limit=0.75, velocity_limit=2)
    with MujocoHardware(model, joints={"hinge": config, "slide": config}) as hw:
        hw.set_positions({"hinge": 0.8, "slide": 0.4})
        state = hw.step()
        assert state.joints["hinge"].effort == pytest.approx(0.75)
        assert state.joints["slide"].effort == pytest.approx(0.75)
        np.testing.assert_allclose(hw.data.qfrc_applied, [0.75, 0.75])


@pytest.mark.parametrize("position, effort", [(1.0, 1.0), (-1.0, -1.0)])
def test_driver_removes_effort_pointing_out_of_joint_limit(hardware, position, effort):
    jid = mujoco.mj_name2id(hardware.model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")
    hardware.data.qpos[hardware.model.jnt_qposadr[jid]] = position
    mujoco.mj_forward(hardware.model, hardware.data)
    hardware.set_efforts({"hinge": effort})
    assert hardware.step().joints["hinge"].effort == pytest.approx(0)
    hardware.set_efforts({"hinge": -effort})
    state = hardware.step()
    assert state.joints["hinge"].effort == pytest.approx(-effort)


def test_free_joint_does_not_shift_scalar_state_or_force_address():
    model = mujoco.MjModel.from_xml_string(FLOATING_XML)
    with MujocoHardware(model) as hw:
        assert hw.joint_names == ("hinge",)
        assert hw.controlled_joints == ("hinge",)
        hw.data.qpos[7] = 0.35
        hw.data.qvel[6] = 0.7
        mujoco.mj_forward(model, hw.data)
        state = hw.read()
        assert state.joints["hinge"].position == pytest.approx(0.35)
        assert state.joints["hinge"].velocity == pytest.approx(0.7)
        hw.set_efforts({"hinge": 0.3})
        hw.step()
        np.testing.assert_allclose(hw.data.qfrc_applied[:6], 0)
        assert hw.data.qfrc_applied[6] == pytest.approx(0.3)


def test_passive_joints_can_be_read_but_cannot_be_commanded(model):
    with MujocoHardware(model, joints={"hinge": JointConfig()}) as hw:
        assert hw.controlled_joints == ("hinge",)
        assert hw.joint_info["slide"].controllable is False
        assert hw.read().joints["slide"].mode == "disabled"
        for method in (hw.set_positions, hw.set_velocities, hw.set_efforts):
            with pytest.raises(ValueError):
                method({"slide": 0.1})


@pytest.mark.parametrize("method", ["set_positions", "set_velocities", "set_efforts"])
@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_nonfinite_commands_are_rejected(hardware, method, bad_value):
    with pytest.raises(ValueError):
        getattr(hardware, method)({"hinge": bad_value})


@pytest.mark.parametrize(
    "method, commands",
    [
        ("set_positions", {"hinge": -0.3, "slide": 0.6}),
        ("set_positions", {"hinge": -0.3, "missing": 0}),
        ("set_velocities", {"hinge": -0.3, "slide": 2.1}),
        ("set_efforts", {"hinge": -0.3, "slide": 100.1}),
    ],
)
def test_invalid_batch_leaves_every_previous_command_intact(hardware, method, commands):
    hardware.set_positions({"hinge": 0.6, "slide": 0.2})
    with pytest.raises(ValueError):
        getattr(hardware, method)(commands)
    state = hardware.step(1000)
    assert state.joints["hinge"].mode == "position"
    assert state.joints["hinge"].position == pytest.approx(0.6, abs=0.005)
    assert state.joints["slide"].position == pytest.approx(0.2, abs=0.005)


@pytest.mark.parametrize("steps", [-1, 1.5, True, False, math.nan])
def test_invalid_step_counts_do_not_advance_time(hardware, steps):
    with pytest.raises((TypeError, ValueError)):
        hardware.step(steps)
    assert hardware.read().time == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kp": -1}, {"kp": math.nan}, {"kd": -1}, {"kd": math.inf},
        {"effort_limit": 0}, {"effort_limit": math.inf},
        {"velocity_limit": 0}, {"velocity_limit": math.nan},
    ],
)
def test_invalid_drive_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        JointConfig(**kwargs)


def test_custom_timestep_and_reset_positions(model):
    with MujocoHardware(model, timestep=0.005) as hw:
        assert hw.step(4).time == pytest.approx(0.02)
        hw.reset({"hinge": -0.2, "slide": 0.3})
        state = hw.read()
        assert state.time == 0
        assert state.joints["hinge"].position == pytest.approx(-0.2)
        assert state.joints["slide"].position == pytest.approx(0.3)
        assert state.joints["hinge"].velocity == 0
        state = hw.step(20)
        assert state.joints["hinge"].position == pytest.approx(-0.2, abs=1e-10)
        hw.reset()
        assert hw.read().joints["hinge"].position == pytest.approx(0.1)


def test_invalid_reset_is_atomic(hardware):
    hardware.set_efforts({"hinge": 0.2})
    before = hardware.step(10)
    qpos = hardware.data.qpos.copy()
    qvel = hardware.data.qvel.copy()
    with pytest.raises(ValueError):
        hardware.reset({"hinge": -0.2, "slide": 99})
    after = hardware.read()
    assert after.time == before.time
    assert after.joints["hinge"].mode == before.joints["hinge"].mode
    np.testing.assert_array_equal(hardware.data.qpos, qpos)
    np.testing.assert_array_equal(hardware.data.qvel, qvel)


def test_stop_immediately_removes_drive_and_reset_preserves_stop_lock(hardware):
    hardware.set_positions({"hinge": 0.8})
    hardware.step()
    assert np.any(hardware.data.qfrc_applied != 0)
    hardware.stop()
    assert hardware.read().stopped is True
    np.testing.assert_array_equal(hardware.data.qfrc_applied, 0)
    for method in (hardware.set_positions, hardware.set_velocities, hardware.set_efforts):
        with pytest.raises(RuntimeError):
            method({"hinge": 0})
    hardware.step(10)  # A drive stop does not suspend the physical simulation.
    np.testing.assert_array_equal(hardware.data.qfrc_applied, 0)
    hardware.reset()
    assert hardware.read().stopped is True
    with pytest.raises(RuntimeError):
        hardware.set_positions({"hinge": 0.4})
    hardware.clear_stop()
    assert hardware.read().stopped is False
    state = hardware.step(100)
    # The target from before the stop must never reappear after clearing it.
    assert state.joints["hinge"].position == pytest.approx(0.1, abs=1e-10)
    assert state.joints["hinge"].mode == "position"


def test_readings_and_sensor_arrays_are_snapshots(hardware):
    state = hardware.read()
    sensors = hardware.read_sensors()
    position, quaternion = hardware.body_pose("arm")
    assert sensors["hinge_position"][0] == pytest.approx(0.1)
    np.testing.assert_allclose(position, [0, 0, 1])
    assert np.linalg.norm(quaternion) == pytest.approx(1)
    sensors["hinge_position"][:] = 99
    position[:] = 99
    quaternion[:] = 99
    hardware.set_positions({"hinge": 0.5})
    hardware.step(100)
    assert state.time == 0
    assert state.joints["hinge"].position == pytest.approx(0.1)
    assert hardware.read_sensors()["hinge_position"][0] != 99
    new_position, new_quaternion = hardware.body_pose("arm")
    np.testing.assert_allclose(new_position, [0, 0, 1])
    assert np.linalg.norm(new_quaternion) == pytest.approx(1)


def test_close_and_context_manager_release_api(model):
    with MujocoHardware(model) as hw:
        hw.step()
    for operation in (
        hw.read, hw.read_sensors, hw.step, hw.reset, hw.stop, hw.clear_stop,
        lambda: hw.body_pose("arm"), lambda: hw.set_positions({"hinge": 0}),
    ):
        with pytest.raises(RuntimeError):
            operation()
    hw.close()  # Closing twice is safe.
