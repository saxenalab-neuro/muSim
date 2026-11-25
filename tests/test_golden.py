"""Golden-output tests: today's results must come out exactly the same.

How it works: training with a fixed random seed (123456) on CPU gives exactly
the same numbers every run. So we ran muSim once, saved what it produced in
tests/golden/ (see make_golden.py), and these tests re-run it and require
identical output. If a refactor changes behaviour by accident, even slightly,
these tests fail and say where.

What they cover:
  - adding the targets to the arm model
  - four short training runs (the default setup, plus three that switch on
    features the default leaves off; see TRAIN_VARIANTS in musim_runner.py)
  - the lab's trained model, run on all 6 conditions

If a change is MEANT to alter behaviour (e.g. a bug fix), regenerate the
saved results with `python tests/make_golden.py` and say so in the commit.

Limit: they protect today's behaviour, bugs included. They show that
something changed, not whether muSim is correct. Exact matching also assumes
the same kind of machine and library versions (see tests/README.md).
"""
import json

import numpy as np
import pytest

import musim_runner as mr


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    """A temporary folder, the default config and the arm model with targets.
    Built once and shared by every test in this file."""
    tmp = tmp_path_factory.mktemp("musim")
    cfg = mr.write_config(tmp)
    return tmp, cfg, mr.build_model(tmp, cfg)


def test_append_targets_matches_golden(setup):
    """Adding target bodies to the arm model gives the same file, character for character."""
    _, _, model = setup
    made = (model.parent / "musculo_targets.xml").read_text()
    assert made == (mr.GOLDEN / "musculo_targets.xml").read_text()


@pytest.fixture(scope="module", params=list(mr.TRAIN_VARIANTS))
def train_run(setup, request):
    """One short training run per setup in TRAIN_VARIANTS: (setup name, output folder).

    pytest runs every test that uses this fixture once per setup, and each
    training run happens only once (about 30 s each).
    """
    return request.param, mr.train(*setup, request.param)


def test_training_stats_match_golden(train_run):
    """Every episode's reward, step count and losses match the saved run exactly."""
    variant, folder = train_run
    golden = np.load(mr.GOLDEN / f"train_{variant}_stats.npz")
    stats = mr.load_stats(folder)
    for name in mr.STAT_NAMES:
        a, b = golden[name], stats[name]
        assert a.shape == b.shape, f"{name}: {b.shape} episodes, golden has {a.shape}"
        # Exact comparison. NaN never equals NaN, so count "both NaN" as matching
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        if not same.all():
            # Report the first episode that differs, which is where to start looking
            ep = int(np.argmin(same))
            pytest.fail(f"{name} first differs at episode {ep}: {b[ep]!r} vs golden {a[ep]!r}")


@pytest.mark.parametrize("ckpt", ["agent_networks", "agent_networks_best"])
def test_trained_weights_match_golden(train_run, ckpt):
    """The saved network weights match exactly, in both checkpoint files
    (the latest one, and the best one by reward). Compared by fingerprint."""
    variant, folder = train_run
    golden = json.loads((mr.GOLDEN / f"train_{variant}_checkpoints.json").read_text())[ckpt]
    assert mr.checkpoint_summary(folder / "checkpoint" / f"{ckpt}.pth") == golden


@pytest.fixture(scope="module")
def pretrained(setup):
    """The lab's trained model run in test mode once, summarised."""
    return mr.summarize_test(mr.run_pretrained(*setup))


def test_pretrained_tracking_error_matches_golden(pretrained):
    """Mean hand-target distance (cm) per condition is unchanged.

    Checked separately from the fingerprints below so that, if tracking
    changes, the failure shows a readable number."""
    golden = json.loads((mr.GOLDEN / "pretrained_test.json").read_text())
    assert pretrained["mean_error_cm"] == golden["mean_error_cm"]


def test_pretrained_outputs_match_golden(pretrained):
    """Muscle commands, RNN activity and inputs, and the hand and target paths
    are unchanged for every condition. Compared by fingerprint."""
    golden = json.loads((mr.GOLDEN / "pretrained_test.json").read_text())
    for name in mr.TEST_ARRAYS:
        assert pretrained["hashes"][name] == golden["hashes"][name], f"{name} changed"
