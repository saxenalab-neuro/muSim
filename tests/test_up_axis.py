"""Tests for which direction in the kinematics data is "up".

Why this file exists:
    MuJoCo's "up" is z. The cycling data was recorded with y as "up". The code
    used to deal with that silently, for all data: append_musculo_targets.py
    wired the data's y to MuJoCo's z axis (and the data's z to MuJoCo's y). So
    data that already had z as "up" came out sideways.

    Now the default is "z is up": x, y and z in the data go to MuJoCo's x, y
    and z. For data with y as "up", the config sets

        kinematics_up_axis = y

    and config.load_kinematics() swaps the data's y and z when it loads the
    file. config.to_model_axes() does the same swap for a single [x, y, z]
    triple, such as a target's `center` from the config.

These tests use a tiny made-up kinematics file, so no MuJoCo model is built.
The end-to-end check (z-up data behaves exactly like the cycling data) is in
test_generalisation.py (test_dataset_with_z_up_needs_no_setting).

Run from the repository root:
    python -m pytest tests/test_up_axis.py
"""

import os
import pickle
import sys

import numpy as np
import pytest

# Make the repository root importable, so `import config` finds config.py.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import config  # noqa: E402  (import after the path is set up)

LAB_CONFIG = os.path.join(REPO_ROOT, "configs", "configs.txt")

# Easy-to-spot values for each row of the made-up data.
X, Y, Z = 0.11, 0.22, 0.33


def write_kinematics(folder):
    """Save a tiny kinematics.pkl: 1 target, rows x/y/z = X/Y/Z, 4 time points,
    one training movement and one test movement."""
    movement = np.array([[np.full(4, X), np.full(4, Y), np.full(4, Z)]])   # [1, 3, 4]
    kin = {"marker_names": ["hand"], "train": {0: movement.copy()}, "test": {0: movement.copy()}}
    with open(os.path.join(folder, "kinematics.pkl"), "wb") as f:
        pickle.dump(kin, f)


def load_with(folder, *cli_args):
    """Parse the settings in cli_args and load the tiny file with them.
    Returns (args, the loaded data)."""
    args = config.config_parser().parse_args(["--kinematics_path", str(folder), *cli_args])
    return args, config.load_kinematics(args)


def rows(movement):
    """The value held by rows 0, 1 and 2 of the first target."""
    return [movement[0, r, 0] for r in range(3)]


def test_default_is_z_up_and_leaves_the_data_alone(tmp_path):
    """With no setting, z is "up": the data comes back exactly as saved."""
    write_kinematics(tmp_path)
    args, kin = load_with(tmp_path)
    assert args.kinematics_up_axis == "z"
    assert rows(kin["train"][0]) == [X, Y, Z]
    assert rows(kin["test"][0]) == [X, Y, Z]


def test_y_up_swaps_y_and_z_in_train_and_test(tmp_path):
    """With kinematics_up_axis = y, the file's y ends up on row 2 (MuJoCo's z,
    up) and the file's z on row 1. x and the marker names are untouched."""
    write_kinematics(tmp_path)
    _, kin = load_with(tmp_path, "--kinematics_up_axis", "y")
    assert rows(kin["train"][0]) == [X, Z, Y]
    assert rows(kin["test"][0]) == [X, Z, Y]
    assert kin["train"][0].shape == (1, 3, 4)
    assert kin["marker_names"] == ["hand"]


def test_center_gets_the_same_swap(tmp_path):
    """A target's `center` is written in the data's own x, y, z, so it is
    swapped exactly like the data."""
    args_z = config.config_parser().parse_args([])
    args_y = config.config_parser().parse_args(["--kinematics_up_axis", "y"])
    assert config.to_model_axes([0.06, 0.083, 0], args_z) == [0.06, 0.083, 0]
    assert config.to_model_axes([0.06, 0.083, 0], args_y) == [0.06, 0, 0.083]


def test_only_y_or_z_is_accepted():
    """A typo such as kinematics_up_axis = x stops the run straight away,
    instead of quietly being treated as z."""
    with pytest.raises(SystemExit):
        config.config_parser().parse_args(["--kinematics_up_axis", "x"])


def test_lab_config_says_the_cycling_data_has_y_up():
    """The cycling data has y as "up", so the lab's config has to say so.
    Without this line the cycling target would move sideways."""
    args = config.config_parser().parse_args(["--config", LAB_CONFIG])
    assert args.kinematics_up_axis == "y"
