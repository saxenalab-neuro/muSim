"""What muSim should handle next. These tests FAIL on today's code, on purpose.

Each test describes one thing muSim can't do yet: a dataset without neural
recordings, or movements with other lengths. Each builds a small custom setup
with scenarios.py.

How "expected to fail" works:
  - Each test is marked @pytest.mark.xfail(strict=True, ...). pytest reports
    it as "xfailed" (expected failure), and the test run still passes.
  - It only counts as expected if it fails with the stated error type
    (raises=AssertionError: muSim ran, but gave the wrong result). Any other
    failure, such as a mistake in the test setup, shows up as a real failure.
  - When a fix makes a test pass, pytest reports XPASS(strict) and the run
    FAILS. That's the reminder to delete the test's xfail line and move it to
    test_generalisation.py, where it becomes an ordinary test that protects
    the fix. Five tests have already moved there.

IDs (H2, B1, ...) refer to the hardcodes and bugs in the codebase guide.
"""
import numpy as np
import pytest

from scenarios import Scenario, cycling_data


# --- Control: the scenario builder itself works on the unchanged cycling setup ---

def test_scenario_builder_on_cycling_data(tmp_path):
    """An ordinary test (should pass). Builds the unchanged cycling setup with
    scenarios.py and checks it behaves normally. So we know the failures below,
    and the passes in test_generalisation.py, come from muSim and not from the
    way we build setups."""
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
