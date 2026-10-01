"""Command line runner for the MuJoCo hardware interface."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from enum import Enum
import json
import math
import os
import sys
import time

from .models import available_robots, available_scenes


def _command(value: str) -> tuple[str, float]:
    try:
        name, text = value.split("=", 1)
        position = float(text)
        if not name or not math.isfinite(position):
            raise ValueError
        return name, position
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected JOINT=FINITE_NUMBER in SI units") from error


def _json_default(value):
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a robot through RoboArena's MuJoCo hardware interface.")
    parser.add_argument("--robot", default="panda", help="robot name (default: panda)")
    parser.add_argument("--scene", default="empty.xml", help="scene filename (default: empty.xml)")
    parser.add_argument("--model", metavar="XML", help="load an arbitrary MJCF file instead of a registered robot")
    parser.add_argument("--list-robots", action="store_true", help="list registered robots and scenes without loading MuJoCo")
    parser.add_argument("--list-joints", action="store_true", help="print joint metadata as JSON and exit")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--headless", action="store_true", help="simulate without a window and print JSON state")
    display.add_argument("--viewer", action="store_true", help="open a viewer (default)")
    parser.add_argument("--steps", type=int, help="number of physics steps (headless default: 1000; viewer default: unlimited)")
    parser.add_argument("--timestep", type=float, help="override model timestep in seconds")
    for mode, units in (("position", "rad or m"), ("velocity", "rad/s or m/s"), ("effort", "Nm or N")):
        parser.add_argument(f"--{mode}", action="append", type=_command, default=[], metavar="JOINT=VALUE", help=f"set a joint {mode} target ({units}); may be repeated")
    parser.add_argument("--gravity-compensation", action="store_true", help="compensate model bias forces (gravity and Coriolis) in joint control")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.steps is not None and arguments.steps < 0:
        parser.error("--steps must be nonnegative")
    if arguments.timestep is not None and (
        not math.isfinite(arguments.timestep) or arguments.timestep <= 0
    ):
        parser.error("--timestep must be finite and positive")

    seen: set[str] = set()
    for mode in ("position", "velocity", "effort"):
        for name, _ in getattr(arguments, mode):
            if name in seen:
                parser.error(f"Joint {name!r} has more than one command; choose a single mode and target")
            seen.add(name)

    try:
        if arguments.list_robots:
            robots = {robot: available_scenes(robot) for robot in available_robots()}
            print(json.dumps(robots, indent=2, ensure_ascii=False))
            return 0

        # Delayed imports let --list-robots work without a rendering backend.
        from .hardware import MujocoHardware

        options = {"gravity_compensation": arguments.gravity_compensation}
        if arguments.timestep is not None:
            options["timestep"] = arguments.timestep
        hardware = (
            MujocoHardware.from_xml(arguments.model, **options)
            if arguments.model
            else MujocoHardware.from_robot(arguments.robot, arguments.scene, **options)
        )
        with hardware:
            if arguments.list_joints:
                metadata = {name: asdict(info) for name, info in hardware.joint_info.items()}
                print(json.dumps(metadata, indent=2, ensure_ascii=False, default=_json_default))
                return 0
            if arguments.position:
                hardware.set_positions(dict(arguments.position))
            if arguments.velocity:
                hardware.set_velocities(dict(arguments.velocity))
            if arguments.effort:
                hardware.set_efforts(dict(arguments.effort))
            if arguments.headless:
                steps = arguments.steps if arguments.steps is not None else 1000
                hardware.step(steps)
                print(json.dumps(asdict(hardware.read()), indent=2, ensure_ascii=False, default=_json_default))
                return 0
            if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
                raise RuntimeError("No display is available. Use --headless to run without a viewer.")

            import mujoco.viewer

            with mujoco.viewer.launch_passive(hardware.model, hardware.data) as viewer:
                count = 0
                deadline = time.monotonic()
                while viewer.is_running() and (arguments.steps is None or count < arguments.steps):
                    # The controller applies forces and advances the real model;
                    # position holding never freezes qpos or qvel.
                    with viewer.lock():
                        hardware.step()
                    viewer.sync()
                    count += 1
                    deadline += hardware.model.opt.timestep
                    delay = deadline - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                    elif delay < -0.25:
                        deadline = time.monotonic()
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f"roboarena: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
