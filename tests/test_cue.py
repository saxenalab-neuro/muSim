"""Tests for the cue file (cue.pkl).

Why this file exists:
    The cue is the number that tells the network which movement (condition)
    it is doing. It is the network's last input at every step.

    The code used to work the cue out by itself from the movement's length,
    with two numbers from the cycling data typed in:

        cue = (time points - 600) / (1319 - 600)

    Now the user gives the cue in its own file, cue.pkl, in the folder named
    by the config setting `cue_data_path`. It has the same layout as
    kinematics.pkl ('train' and 'test', one entry per movement), and each cue
    has one value per time point, so it is as long as that movement's
    kinematics. config.load_cue() loads it and checks it against the
    kinematics, and the environment reads the cue for the current time point.

The first tests call config.load_cue() on tiny made-up files, so no MuJoCo
model is built. The last one uses the real environment (the `env` fixture
from conftest.py).

Run from the repository root:
    python -m pytest tests/test_cue.py
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

# A tiny made-up dataset: 2 training movements (5 and 8 time points) and 1
# test movement (6 time points), each [1 target, 3 directions, time points].
KINEMATICS = {
    "marker_names": ["hand"],
    "train": {0: np.zeros((1, 3, 5)), 1: np.zeros((1, 3, 8))},
    "test": {0: np.zeros((1, 3, 6))},
}


def good_cue():
    """A cue that fits KINEMATICS: one value per time point of each movement."""
    return {"train": {0: np.full(5, 0.25), 1: np.full(8, 0.75)}, "test": {0: np.full(6, 0.5)}}


def load(folder, cue):
    """Save `cue` as cue.pkl in `folder` (skipped if cue is None) and load it
    the way muSim does, checking it against KINEMATICS."""
    if cue is not None:
        with open(os.path.join(folder, "cue.pkl"), "wb") as f:
            pickle.dump(cue, f)
    args = config.config_parser().parse_args(["--cue_data_path", str(folder)])
    return config.load_cue(args, KINEMATICS)


def test_a_cue_that_fits_is_loaded_as_it_is(tmp_path):
    """A cue with one value per time point comes back unchanged."""
    cue = load(tmp_path, good_cue())
    np.testing.assert_array_equal(cue["train"][0], np.full(5, 0.25))
    np.testing.assert_array_equal(cue["train"][1], np.full(8, 0.75))
    np.testing.assert_array_equal(cue["test"][0], np.full(6, 0.5))


def test_missing_file_says_what_to_provide(tmp_path):
    """No cue.pkl: the run stops and names the missing file."""
    with pytest.raises(FileNotFoundError, match="cue.pkl"):
        load(tmp_path, None)


def test_cue_of_the_wrong_length_is_rejected(tmp_path):
    """A cue with 7 values for a movement with 8 time points stops the run.
    The error names the movement and both lengths."""
    cue = good_cue()
    cue["train"][1] = np.full(7, 0.75)
    with pytest.raises(ValueError, match=r"'train' movement 1.*\(7,\).*8 time points"):
        load(tmp_path, cue)


def test_one_number_per_movement_is_rejected(tmp_path):
    """A single number instead of one value per time point stops the run."""
    cue = good_cue()
    cue["test"][0] = 0.5
    with pytest.raises(ValueError, match="'test' movement 0"):
        load(tmp_path, cue)


def test_a_movement_without_a_cue_is_rejected(tmp_path):
    """The cue file must cover exactly the movements in the kinematics."""
    cue = good_cue()
    del cue["train"][1]
    with pytest.raises(ValueError, match="must match"):
        load(tmp_path, cue)


def test_a_file_without_test_cues_is_rejected(tmp_path):
    """The file needs both 'train' and 'test', like kinematics.pkl."""
    cue = good_cue()
    del cue["test"]
    with pytest.raises(ValueError, match="no 'test' entry"):
        load(tmp_path, cue)


def test_lab_cue_file_holds_the_old_formula():
    """The lab's cue_data/cue.pkl gives the cycling data exactly the numbers
    the code used to work out: (T - 600) / (1319 - 600) at every time point.
    So the cycling results are unchanged."""
    args = config.config_parser().parse_args(["--config", LAB_CONFIG])
    cue = config.load_cue(args, config.load_kinematics(args))
    lengths = {"train": [702, 803, 932, 1319], "test": [619, 1106]}
    for split, Ts in lengths.items():
        for cond, T in enumerate(Ts):
            np.testing.assert_array_equal(cue[split][cond], np.full(T, (T - 600) / (1319 - 600)))


def test_cue_can_change_during_a_movement(env):
    """The network gets the cue for the current time point, so a cue that
    changes during a movement is followed step by step."""
    cond = 1
    T = env.kin_to_sim[cond].shape[-1]
    env.cue_to_sim[cond] = np.linspace(0.0, 1.0, T)     # rises from 0 to 1
    rest = np.zeros(env.model.nu)                        # every muscle command at 0

    env.reset(cond)
    assert env.condition_scalar == 0.0                   # the first time point

    env.istep = 124
    env.step(rest)                                       # 100 steps into the movement
    t = env.tpoint_to_sim
    assert t == int((T - 1) / T * 100)
    assert env.condition_scalar == np.linspace(0.0, 1.0, T)[t]
