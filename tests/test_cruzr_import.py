"""Portable model import tests use synthetic assets, never a stand-in Cruzr."""

import importlib.util
import json
from pathlib import Path
import shutil

import pytest

mujoco = pytest.importorskip("mujoco")
module_spec = importlib.util.spec_from_file_location(
    "cruzr_import", Path(__file__).resolve().parents[1] / "scripts/import_cruzr_s2.py"
)
importer = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(importer)


@pytest.fixture
def source_package(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "tetra.obj").write_text(
        "v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nv 0 0 0.1\n"
        "f 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
    )
    (source / "model.urdf").write_text('''
<robot name="synthetic_import_fixture">
  <link name="base"/>
  <link name="tip">
    <inertial><mass value="1"/>
      <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <visual><geometry><mesh filename="package://fixture/tetra.obj"/></geometry></visual>
    <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>
  </link>
  <joint name="axis" type="revolute">
    <parent link="base"/><child link="tip"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="3" velocity="2"/>
  </joint>
</robot>
''')
    (source / "LICENSE").write_text("Synthetic fixture, CC0\n")
    return source


def test_import_is_portable_and_keeps_provenance(source_package, tmp_path):
    scene = importer.import_model(
        source_package / "model.urdf", [source_package / "LICENSE"],
        tmp_path / "assets", {"fixture": source_package},
    )
    shutil.rmtree(source_package)
    moved = tmp_path / "relocated"
    (tmp_path / "assets").rename(moved)
    model = mujoco.MjModel.from_xml_path(str(moved / "scenes/cruzr_s2/empty.xml"))
    assert model.njnt == 1
    assert model.nmesh == 1
    assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, 0) == "axis"
    metadata = json.loads((moved / "robots/cruzr_s2/provenance.json").read_text())
    assert metadata["source_format"] == "URDF"
    assert metadata["joints"] == ["axis"]
    assert len(metadata["dependencies"][0]["sha256"]) == 64
    assert (moved / "robots/cruzr_s2" / metadata["licenses"][0]["file"]).is_file()


def test_existing_installation_is_preserved(source_package, tmp_path):
    destination = tmp_path / "assets"
    existing = destination / "robots/cruzr_s2"
    existing.mkdir(parents=True)
    sentinel = existing / "user-edits.txt"
    sentinel.write_text("preserve me")
    with pytest.raises(ValueError, match="overwrite"):
        importer.import_model(
            source_package / "model.urdf", [source_package / "LICENSE"],
            destination, {"fixture": source_package},
        )
    assert sentinel.read_text() == "preserve me"


@pytest.mark.parametrize("failure", ["missing-mesh", "mimic", "usd", "license"])
def test_failed_import_does_not_publish(source_package, tmp_path, failure):
    source = source_package / "model.urdf"
    license_file = source_package / "LICENSE"
    if failure == "missing-mesh":
        (source_package / "tetra.obj").unlink()
    elif failure == "mimic":
        source.write_text(source.read_text().replace(
            '<parent link="base"/>', '<mimic joint="other"/><parent link="base"/>'
        ))
    elif failure == "usd":
        source = source_package / "model.usda"
        source.write_text("#usda 1.0")
    else:
        license_file.unlink()
    destination = tmp_path / "assets"
    with pytest.raises(Exception):
        importer.import_model(source, [license_file], destination, {"fixture": source_package})
    assert not (destination / "robots/cruzr_s2").exists()
    assert not (destination / "scenes/cruzr_s2").exists()
