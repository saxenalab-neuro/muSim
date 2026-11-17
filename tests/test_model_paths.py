"""Tests for where the model-with-targets file is written and read.

Why this file exists:
    append_musculo_targets.py adds the target bodies to the arm model and saves
    the result as a second model file. Every later script (find_init_pose.py,
    simulate.py, ...) then loads that second file. Its path used to be worked
    out by cutting len('musculoskeletal_model.xml') = 25 characters off the
    model path, so any model with a different file name (e.g. arm.xml) ended
    up in the wrong folder or crashed.

    Now config.get_musculo_targets_path() decides, in one place:
      - the config setting `musculo_targets_path`, if given;
      - otherwise musculo_targets.xml next to the model.

Run from the repository root:
    python -m pytest tests/test_model_paths.py
"""

import os
import shutil
import subprocess
import sys
from xml.etree import ElementTree as ET

# Make the repository root importable, so `import config` finds config.py.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import config  # noqa: E402  (import after the path is set up)

LAB_MODEL = os.path.join(REPO_ROOT, "musculoskeletal_model", "musculoskeletal_model.xml")
LAB_CONFIG = os.path.join(REPO_ROOT, "configs", "configs.txt")


def targets_path_for(*cli_args):
    """Parse the lab config plus extra arguments; return the targets path."""
    args = config.config_parser().parse_args(["--config", LAB_CONFIG, *cli_args])
    return config.get_musculo_targets_path(args)


def test_lab_setup_is_unchanged():
    """The lab's config gives the same file as before the fix."""
    assert targets_path_for() == os.path.join("musculoskeletal_model", "musculo_targets.xml")


def test_any_model_name_puts_it_next_to_the_model():
    """mouse/arm.xml -> mouse/musculo_targets.xml (no cutting of letters).

    The old code cut 25 letters off "mouse/arm.xml" (13 letters), leaving
    nothing, so the file landed in the main muSim folder instead.
    """
    path = targets_path_for("--musculoskeletal_model_path", os.path.join("mouse", "arm.xml"))
    assert path == os.path.join("mouse", "musculo_targets.xml")


def test_setting_overrides_the_default():
    """If the config gives musculo_targets_path, exactly that is used."""
    path = targets_path_for("--musculoskeletal_model_path", os.path.join("mouse", "arm.xml"),
                            "--musculo_targets_path", os.path.join("out", "mouse_targets.xml"))
    assert path == os.path.join("out", "mouse_targets.xml")


def test_add_targets_script_writes_next_to_arm_xml(tmp_path):
    """End to end: run append_musculo_targets.py on mouse/arm.xml.

    Runs the script the way a user would, and checks the new file appears
    next to the model, with the target ball added.
    """
    model_dir = tmp_path / "mouse"
    model_dir.mkdir()
    model = model_dir / "arm.xml"
    shutil.copy(LAB_MODEL, model)

    result = subprocess.run(
        [sys.executable, "append_musculo_targets.py",
         "--config", LAB_CONFIG,
         "--musculoskeletal_model_path", str(model)],
        cwd=REPO_ROOT, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    output = model_dir / "musculo_targets.xml"
    assert output.is_file(), "the model with targets was not written next to arm.xml"
    bodies = [b.get("name") for b in ET.parse(output).iter("body")]
    assert "target0" in bodies
