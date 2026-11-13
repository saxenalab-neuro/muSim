"""Setups other than the lab's cycling one that muSim now handles.

Each test here used to fail on purpose (it lived in test_future.py with an
"expected to fail" label), because muSim had a value fixed to the cycling
setup. Each one now passes because of a specific fix, and stays here as an
ordinary test, so the fix can't be undone by accident.

Each test builds a small custom setup with scenarios.py (a renamed body part,
a second marker, a different locked axis, a shorter config, a model file with
another name) and runs the real muSim code on it.

The control test that checks scenarios.py itself on the unchanged cycling
setup is in test_future.py (test_scenario_builder_on_cycling_data).

IDs (H1, B2, ...) refer to the hardcodes and bugs in the codebase guide.
"""
import numpy as np

from scenarios import Scenario, cycling_data


def two_targets_args(n):
    """Command-line settings for n markers. trajectory_scaling and center are
    given once per marker, so they have to be repeated n times."""
    return ["--trajectory_scaling", *["26.3157894737"] * n, "--center", *["[0.06,0.083,0]"] * n]


# --- Choose the body part that tracks the target --------------------------------

def test_end_effector_with_any_name(tmp_path):
    """Training works when the tracked body is called "paw" instead of "hand".

    Used to fail (H1): is_done() looked up a body named "hand", so MuJoCo
    raised "Invalid name 'hand'". Fixed by reading the tracked body names from
    the data (marker_names).
    """
    kin, neural = cycling_data()
    kin["marker_names"] = ["paw"]
    s = Scenario(tmp_path, kin, neural, rename_hand="paw",
                 extra_args=["--visual_distance_bodies", "[paw;target0]"])
    steps = s.train()
    assert len(steps) == 3


def test_every_marker_can_end_episode(tmp_path):
    """With two markers, the second one leaving its zone ends the episode.

    Used to fail (H1): is_done() only checked "hand" against "target0". The
    second marker got the -25 penalty, but the episode kept going. Fixed by
    checking every body/target pair from the data, as the reward does.
    """
    kin, neural = cycling_data()
    kin["marker_names"] = ["hand", "radius"]
    for split in ["train", "test"]:
        for c, k in kin[split].items():
            # Add a second marker: a copy of the hand path shifted far in x, so
            # its target sits ~0.5 m from the forearm after scaling (well outside its zone)
            kin[split][c] = np.concatenate([k, k + np.array([[[13.0], [0], [0]]])])
    r = Scenario(tmp_path, kin, neural, extra_args=two_targets_args(2)).probe()
    # Make sure the setup did what we intended before judging muSim
    if r["rewards"][1] != -25:
        raise RuntimeError(f"setup: expected the second marker out of its zone, got {r}")
    assert r["dones"][1], "second marker left its zone but the episode kept going"


# --- Datasets that differ from the cycling data --------------------------------

def test_dataset_with_x_locked(tmp_path):
    """Tracking y and z, with x locked (NaN), puts the target where the data says.

    Used to fail (B2): the y target read data row "number of tracked axes
    before y", which is row 0 (x's NaN row) when x is locked. Fixed by always
    reading x, y and z from rows 0, 1 and 2.
    """
    kin, neural = cycling_data()
    for split in ["train", "test"]:
        for c, k in kin[split].items():           # move the path into y/z, lock x
            kin[split][c] = np.stack([np.full_like(k[:, 0], np.nan), k[:, 0], k[:, 1]], axis=1)
    r = Scenario(tmp_path, kin, neural).probe()
    start = r["kin_start"][0]
    assert r["target_qpos"] == {"box:y0": start[1], "box:z0": start[2]}


# --- Settings and models from the config ----------------------------------------

def test_config_can_leave_out_optional_feedback(tmp_path):
    """A config that leaves out the optional visual_velocity and
    visual_distance_bodies settings still works.

    Used to fail (B6): they defaulted to None, and the code calls len() on
    them, which crashed with "'NoneType' has no len()". Fixed by defaulting
    them to an empty list.
    """
    kin, neural = cycling_data()
    s = Scenario(tmp_path, kin, neural, drop_settings=("visual_velocity", "visual_distance_bodies"))
    r = s.probe()
    assert len(r["rewards"]) == 2


def test_model_file_with_any_name(tmp_path):
    """A model file called arm.xml (in a differently named folder) works.

    Used to fail (H6): the scripts built the targets file's path by cutting
    len('musculoskeletal_model.xml') characters off the model path, so any
    other name gave a broken path, and MuJoCo reported "Error opening file".
    Fixed by config.get_musculo_targets_path(), which puts the targets file
    next to the model.
    """
    kin, neural = cycling_data()
    s = Scenario(tmp_path, kin, neural, model_file="arm.xml", model_subdir="monkey_arm_model_folder")
    r = s.probe()
    assert len(r["rewards"]) == 2
