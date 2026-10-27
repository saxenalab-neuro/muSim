"""What the refactor should make possible. These FAIL on today's code.

Each test is marked xfail(strict=True) and only counts as an expected failure
for the specific known reason. When a fix makes one pass, pytest reports it
as XPASS and the run fails, as a reminder to remove its xfail marker so it
becomes an ordinary test. IDs (H1, B1, ...) refer to the codebase guide.
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
    return ["--trajectory_scaling", *["26.3157894737"] * n, "--center", *["[0.06,0.083,0]"] * n]


# --- Control: the scenario builder itself works on the unchanged cycling setup ---

def test_scenario_builder_on_cycling_data(tmp_path):
    kin, neural = cycling_data()
    s = Scenario(tmp_path, kin, neural)
    r = s.probe()
    assert r["rewards"][1] > 14.5 and not any(r["dones"])
    assert r["target_qpos"] == {"box:x0": r["kin_start"][0][0], "box:y0": r["kin_start"][0][1]}
    assert all(0 <= c <= 1 for c in r["cond_scalars"])
    assert s.train().mean() > 20


# --- Item 1: choose the body part that tracks the target ------------------------

@pytest.mark.xfail(strict=True, raises=KnownLimitation,
                   reason="H1: episode-end check is hardcoded to a body named 'hand'")
def test_end_effector_with_any_name(tmp_path):
    kin, neural = cycling_data()
    kin["marker_names"] = ["paw"]
    s = Scenario(tmp_path, kin, neural, rename_hand="paw",
                 extra_args=["--visual_distance_bodies", "[paw;target0]"])
    steps = expect_known("Invalid name 'hand'", s.train)
    assert len(steps) == 3


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="H1: only the first marker can end an episode")
def test_every_marker_can_end_episode(tmp_path):
    kin, neural = cycling_data()
    kin["marker_names"] = ["hand", "radius"]
    for split in ["train", "test"]:
        for c, k in kin[split].items():
            # second marker's target sits ~0.5 m from the forearm after scaling
            kin[split][c] = np.concatenate([k, k + np.array([[[13.0], [0], [0]]])])
    r = Scenario(tmp_path, kin, neural, extra_args=two_targets_args(2)).probe()
    if r["rewards"][1] != -25:
        raise RuntimeError(f"setup: expected the second marker out of its zone, got {r}")
    assert r["dones"][1], "second marker left its zone but the episode kept going"


# --- Datasets that differ from the cycling data --------------------------------

@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="B1: without neural data the kinematics are normalised in place (become NaN)")
def test_dataset_without_neural_data_keeps_kinematics(tmp_path):
    kin, _ = cycling_data()
    raw = kin["train"][0][0, :2].copy()
    r = Scenario(tmp_path, kin, neural=None).probe()
    got = np.array(r["kin_train"]["0"])[0, :2]
    np.testing.assert_array_equal(got, raw / 26.3157894737 + np.array([[0.06], [0.083]]))


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="B1: without neural data every episode ends after 2 steps")
def test_training_without_neural_data(tmp_path):
    kin, _ = cycling_data()
    steps = Scenario(tmp_path, kin, neural=None).train()
    # With the same data plus neural recordings, untrained episodes last ~86 steps
    assert steps.mean() > 20, f"episodes lasted {steps.tolist()} steps"


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="B2: y/z data rows are picked by counting enabled axes")
def test_dataset_with_x_locked(tmp_path):
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
    kin, neural = cycling_data()
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
    kin, neural = cycling_data()
    s = Scenario(tmp_path, kin, neural, drop_settings=("visual_velocity", "visual_distance_bodies"))
    r = expect_known("'NoneType' has no len()", s.probe)
    assert len(r["rewards"]) == 2


@pytest.mark.xfail(strict=True, raises=KnownLimitation,
                   reason="H6: model path is built by cutting off len('musculoskeletal_model.xml')")
def test_model_file_with_any_name(tmp_path):
    kin, neural = cycling_data()
    s = Scenario(tmp_path, kin, neural, model_file="arm.xml", model_subdir="monkey_arm_model_folder")
    r = expect_known("Error opening file", s.probe)
    assert len(r["rewards"]) == 2
