"""What the refactor should make possible. These tests FAIL on today's code, on purpose.

Each test describes one thing items 1 and 2 should allow: a body part not
named "hand", two markers, a dataset without neural recordings, a different
locked axis, other trial lengths, a shorter config, or a model file with
another name. Each builds a small custom setup with scenarios.py.

How "expected to fail" works:
  - Each test is marked @pytest.mark.xfail(strict=True, ...). pytest reports
    it as "xfailed" (expected failure), and the test run still passes.
  - It only counts as expected if it fails for the KNOWN reason: either a
    specific error type (raises=...) or, for crashes, an error message that
    matches the known cause (see expect_known). Any other failure, such as a
    mistake in the test setup, shows up as a real failure.
  - When a fix makes a test pass, pytest reports XPASS(strict) and the run
    FAILS. That's the reminder to delete the test's xfail line so it becomes
    an ordinary test that protects the fix.

IDs (H1, B1, ...) refer to the hardcodes and bugs in the codebase guide.
"""
import numpy as np
import pytest

from scenarios import Scenario, cycling_data


class KnownLimitation(Exception):
    """Today's code failed for the documented reason."""


def expect_known(reason, fn):
    """Run fn. A crash whose error mentions `reason` is the known limitation.
    Any other crash is a real error in the test setup and fails normally."""
    try:
        return fn()
    except RuntimeError as e:
        if reason in str(e):
            raise KnownLimitation(reason) from e
        raise


def two_targets_args(n):
    """Command-line settings for n markers. trajectory_scaling and center are
    given once per marker, so they have to be repeated n times."""
    return ["--trajectory_scaling", *["26.3157894737"] * n, "--center", *["[0.06,0.083,0]"] * n]


# --- Control: the scenario builder itself works on the unchanged cycling setup ---

def test_scenario_builder_on_cycling_data(tmp_path):
    """An ordinary test (should pass). Builds the unchanged cycling setup with
    scenarios.py and checks it behaves normally, so we know the failures below
    come from muSim and not from the way we build setups."""
    kin, neural = cycling_data()
    s = Scenario(tmp_path, kin, neural)
    r = s.probe()
    # Step 2 earns near-maximum reward and the episode continues
    assert r["rewards"][1] > 14.5 and not any(r["dones"])
    # The target starts where the data says
    assert r["target_qpos"] == {"box:x0": r["kin_start"][0][0], "box:y0": r["kin_start"][0][1]}
    # The condition cues are all between 0 and 1
    assert all(0 <= c <= 1 for c in r["cond_scalars"])
    # Untrained episodes last well over 20 steps (about 86 on average)
    assert s.train().mean() > 20


# --- Item 1: choose the body part that tracks the target ------------------------

@pytest.mark.xfail(strict=True, raises=KnownLimitation,
                   reason="H1: episode-end check is hardcoded to a body named 'hand'")
def test_end_effector_with_any_name(tmp_path):
    """Training works when the tracked body is called "paw" instead of "hand".

    Fails today: is_done() looks up a body named "hand", so MuJoCo raises
    "Invalid name 'hand'".
    """
    kin, neural = cycling_data()
    kin["marker_names"] = ["paw"]
    s = Scenario(tmp_path, kin, neural, rename_hand="paw",
                 extra_args=["--visual_distance_bodies", "[paw;target0]"])
    steps = expect_known("Invalid name 'hand'", s.train)
    assert len(steps) == 3


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="H1: only the first marker can end an episode")
def test_every_marker_can_end_episode(tmp_path):
    """With two markers, the second one leaving its zone ends the episode.

    Fails today: is_done() only checks "hand" against "target0". The second
    marker gets the -25 penalty, but the episode keeps going.
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

@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="B1: without neural data the kinematics are normalised in place (become NaN)")
def test_dataset_without_neural_data_keeps_kinematics(tmp_path):
    """A dataset with no neural recordings loads with its hand path intact.

    Fails today: with no neural file, muSim uses the kinematics themselves as a
    placeholder, then "normalises" that placeholder in place. This overwrites
    the real hand path, and because z is NaN, the maximum is NaN, so everything
    becomes NaN.
    """
    kin, _ = cycling_data()
    raw = kin["train"][0][0, :2].copy()
    r = Scenario(tmp_path, kin, neural=None).probe()
    got = np.array(r["kin_train"]["0"])[0, :2]
    # Expected: the normal scaling and centring, as for the cycling data
    np.testing.assert_array_equal(got, raw / 26.3157894737 + np.array([[0.06], [0.083]]))


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="B1: without neural data every episode ends after 2 steps")
def test_training_without_neural_data(tmp_path):
    """Training on a dataset without neural recordings behaves normally.

    Fails today: the target path is all NaN (see the test above), so every
    episode ends after 2 steps, with no error message.
    """
    kin, _ = cycling_data()
    steps = Scenario(tmp_path, kin, neural=None).train()
    # With the same data plus neural recordings, untrained episodes last ~86 steps
    assert steps.mean() > 20, f"episodes lasted {steps.tolist()} steps"


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="B2: y/z data rows are picked by counting enabled axes")
def test_dataset_with_x_locked(tmp_path):
    """Tracking y and z, with x locked (NaN), puts the target where the data says.

    Fails today: the y target reads data row "number of tracked axes before
    y", which is row 0 (x's NaN row) when x is locked. It only works when the
    locked axis is the last one (z, as in the cycling data).
    """
    kin, neural = cycling_data()
    for split in ["train", "test"]:
        for c, k in kin[split].items():           # move the path into y/z, lock x
            kin[split][c] = np.stack([np.full_like(k[:, 0], np.nan), k[:, 0], k[:, 1]], axis=1)
    r = Scenario(tmp_path, kin, neural).probe()
    start = r["kin_start"][0]
    assert r["target_qpos"] == {"box:y0": start[1], "box:z0": start[2]}


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="H2: condition cue uses the cycling data's lengths (600 and 1319)")
def test_condition_cue_scales_to_dataset(tmp_path):
    """The condition cue stays between 0 and 1 for a dataset with shorter trials.

    Fails today: the cue is (T - 600) / (1319 - 600), fixed to the cycling
    data, so trials shorter than 600 samples give negative cues.
    """
    kin, neural = cycling_data()
    # Cut every trial short (300-450 samples, all under 600)
    lengths = {"train": [300, 350, 400, 450], "test": [320, 420]}
    for split, ls in lengths.items():
        for c, n in enumerate(ls):
            kin[split][c] = kin[split][c][..., :n]
            neural[split][c] = neural[split][c][:n]
    cues = Scenario(tmp_path, kin, neural).probe()["cond_scalars"]
    assert all(0 <= c <= 1 for c in cues), f"cues {cues}"


# --- Item 2: settings and models from the config --------------------------------

@pytest.mark.xfail(strict=True, raises=KnownLimitation,
                   reason="B6: leaving out visual_velocity / visual_distance_bodies crashes")
def test_config_can_leave_out_optional_feedback(tmp_path):
    """A config that leaves out the optional visual_velocity and
    visual_distance_bodies settings still works.

    Fails today: they default to None, and the code calls len() on them, which
    crashes with "'NoneType' has no len()".
    """
    kin, neural = cycling_data()
    s = Scenario(tmp_path, kin, neural, drop_settings=("visual_velocity", "visual_distance_bodies"))
    r = expect_known("'NoneType' has no len()", s.probe)
    assert len(r["rewards"]) == 2


@pytest.mark.xfail(strict=True, raises=KnownLimitation,
                   reason="H6: model path is built by cutting off len('musculoskeletal_model.xml')")
def test_model_file_with_any_name(tmp_path):
    """A model file called arm.xml (in a differently named folder) works.

    Fails today: the scripts build the targets file's path by cutting
    len('musculoskeletal_model.xml') characters off the model path, so any
    other name gives a broken path, and MuJoCo reports "Error opening file".
    """
    kin, neural = cycling_data()
    s = Scenario(tmp_path, kin, neural, model_file="arm.xml", model_subdir="monkey_arm_model_folder")
    r = expect_known("Error opening file", s.probe)
    assert len(r["rewards"]) == 2
