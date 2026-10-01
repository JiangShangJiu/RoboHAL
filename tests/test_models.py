import mujoco
import pytest

from roboarena.models import available_robots, available_scenes, load_mjcf, load_robot


@pytest.fixture
def builtin_assets(monkeypatch):
    monkeypatch.delenv("ROBOARENA_ASSETS", raising=False)


def test_builtin_panda_loads_and_steps(builtin_assets):
    model = load_robot("panda", "empty.xml")
    assert model.nq == 9
    assert model.nu == 8
    assert model.joint("joint1").id >= 0
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert data.time == pytest.approx(model.opt.timestep)


def test_scene_discovery_excludes_components_without_scenes(builtin_assets):
    assert "panda" in available_robots()
    assert "allegro" not in available_robots()
    assert "empty.xml" in available_scenes("panda")


def test_external_assets_and_scene_error(tmp_path, monkeypatch):
    scene_dir = tmp_path / "scenes" / "custom"
    scene_dir.mkdir(parents=True)
    (scene_dir / "empty.xml").write_text('<mujoco><worldbody><geom type="sphere" size="0.1"/></worldbody></mujoco>')
    monkeypatch.setenv("ROBOARENA_ASSETS", str(tmp_path))
    assert available_robots() == ["custom"]
    assert available_scenes("custom") == ["empty.xml"]
    assert load_robot("custom").ngeom == 1
    with pytest.raises(FileNotFoundError, match="Available scenes"):
        load_robot("custom", "missing.xml")
    with pytest.raises(ValueError, match="inside its robot directory"):
        load_robot("custom", "../custom/../../other.xml")
    with pytest.raises(ValueError, match="Invalid robot name"):
        load_robot("../custom")


def test_nested_includes_resolve_document_local_meshes_without_changing_sources(tmp_path):
    # Both include documents and mesh files deliberately share basenames.
    # A flat temporary copy would overwrite one of them.
    obj = "v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nv 0 0 0.1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
    sources = {}
    for name in ("left", "right"):
        folder = tmp_path / name
        (folder / "meshes").mkdir(parents=True)
        (folder / "meshes" / "shape.obj").write_text(obj)
        xml = (
            '<mujoco><compiler meshdir="meshes"/>'
            f'<asset><mesh name="{name}" file="shape.obj"/></asset>'
            f'<worldbody><geom type="mesh" mesh="{name}"/></worldbody></mujoco>'
        )
        (folder / "part.xml").write_text(xml)
        sources[folder / "part.xml"] = xml
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / "parts.xml").write_text('<mujoco><include file="../left/part.xml"/><include file="../right/part.xml"/></mujoco>')
    main = tmp_path / "scene.xml"
    main.write_text('<mujoco><include file="nested/parts.xml"/></mujoco>')
    model = load_mjcf(main)
    assert model.nmesh == 2
    assert model.ngeom == 2
    for path, content in sources.items():
        assert path.read_text() == content


def test_inherited_compiler_directory_remains_relative_to_parent(tmp_path):
    (tmp_path / "meshes").mkdir()
    (tmp_path / "nested").mkdir()
    (tmp_path / "meshes" / "tetra.obj").write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
    )
    (tmp_path / "nested" / "asset.xml").write_text(
        '<mujoco><asset><mesh name="tetra" file="tetra.obj"/></asset></mujoco>'
    )
    scene = tmp_path / "scene.xml"
    scene.write_text(
        '<mujoco><compiler meshdir="meshes"/><include file="nested/asset.xml"/>'
        '<worldbody><geom type="mesh" mesh="tetra"/></worldbody></mujoco>'
    )
    assert load_mjcf(scene).nmesh == 1


def test_circular_include_has_actionable_error(tmp_path):
    scene = tmp_path / "scene.xml"
    scene.write_text('<mujoco><include file="scene.xml"/></mujoco>')
    with pytest.raises(ValueError, match="Circular MJCF include"):
        load_mjcf(scene)


def test_missing_include_has_path(tmp_path):
    scene = tmp_path / "scene.xml"
    scene.write_text('<mujoco><include file="missing.xml"/></mujoco>')
    with pytest.raises(FileNotFoundError, match="missing.xml"):
        load_mjcf(scene)


def test_list_cli_works_without_display(monkeypatch, capsys, builtin_assets):
    from roboarena.cli import main

    monkeypatch.delenv("DISPLAY", raising=False)
    assert main(["--list-robots"]) == 0
    assert '"panda"' in capsys.readouterr().out


def test_native_global_compiler_from_include_is_preserved(tmp_path):
    # Standard MJCF allows compiler settings in one include to configure assets
    # in another document. A per-document rewrite must not override this case.
    (tmp_path / "meshes").mkdir()
    (tmp_path / "meshes" / "tetra.obj").write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
    )
    (tmp_path / "compiler.xml").write_text('<mujoco><compiler meshdir="meshes"/></mujoco>')
    scene = tmp_path / "scene.xml"
    scene.write_text(
        '<mujoco><include file="compiler.xml"/>'
        '<asset><mesh name="tetra" file="tetra.obj"/></asset>'
        '<worldbody><geom type="mesh" mesh="tetra"/></worldbody></mujoco>'
    )
    assert mujoco.MjModel.from_xml_path(str(scene)).nmesh == 1
    assert load_mjcf(scene).nmesh == 1


@pytest.fixture
def cli_model(tmp_path):
    path = tmp_path / "joint.xml"
    path.write_text(
        '<mujoco><option gravity="0 0 0" timestep="0.002"/>'
        '<worldbody><body><joint name="axis" type="slide" axis="1 0 0"/>'
        '<geom type="sphere" size="0.1" mass="1"/></body></worldbody></mujoco>'
    )
    return str(path)


@pytest.mark.parametrize("mode,target", [("position", "0.2"), ("velocity", "0.1"), ("effort", "1")])
def test_headless_cli_advances_real_dynamics(cli_model, mode, target, capsys, monkeypatch):
    import json
    from roboarena.cli import main

    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert main(["--model", cli_model, "--headless", "--steps", "500", f"--{mode}", f"axis={target}"]) == 0
    state = json.loads(capsys.readouterr().out)
    assert state["time"] == pytest.approx(1.0)
    joint = state["joints"]["axis"]
    assert joint["mode"] == mode
    assert joint["position"] > 0.01
    if mode == "position":
        assert joint["position"] == pytest.approx(float(target), abs=0.01)
    elif mode == "velocity":
        assert joint["velocity"] == pytest.approx(float(target), abs=0.01)
    else:
        assert joint["effort"] == pytest.approx(float(target))


@pytest.mark.parametrize("commands", [
    ["--position", "axis=0.1", "--velocity", "axis=0.2"],
    ["--effort", "axis=1", "--position", "axis=0.2"],
    ["--velocity", "axis=1", "--velocity", "axis=0.2"],
])
def test_cli_rejects_conflicting_joint_commands(commands, capsys):
    from roboarena.cli import main

    with pytest.raises(SystemExit) as error:
        main(["--headless", *commands])
    assert error.value.code == 2
    assert "more than one command" in capsys.readouterr().err


@pytest.mark.parametrize("arguments", [["--steps", "-1"], ["--position", "axis=nan"], ["--timestep", "0"]])
def test_cli_rejects_invalid_arguments(arguments):
    from roboarena.cli import main

    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2


def test_cli_help_and_list_import_no_mujoco_or_gl(builtin_assets):
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    code = (
        "import sys; from roboarena.cli import main; "
        "main(['--list-robots']); "
        "assert 'mujoco' not in sys.modules; assert 'glfw' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    help_result = subprocess.run([sys.executable, "-m", "roboarena.cli", "--help"], env=env, capture_output=True, text=True, check=True)
    assert "--effort" in help_result.stdout


def test_cli_joint_listing_and_zero_steps(cli_model, capsys):
    import json
    from roboarena.cli import main

    assert main(["--model", cli_model, "--list-joints"]) == 0
    metadata = json.loads(capsys.readouterr().out)
    assert metadata["axis"]["kind"] == "slide"
    assert main(["--model", cli_model, "--headless", "--steps", "0"]) == 0
    assert json.loads(capsys.readouterr().out)["time"] == 0.0

