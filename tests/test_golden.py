""""Nothing changed" tests: today's pipeline must reproduce the saved golden output.

The golden output was made with seed 123456 on CPU. Seeded CPU training is
bit-for-bit repeatable, so these tests compare exactly. If a change is meant
to alter behaviour, regenerate with tests/make_golden.py.
"""
import json

import numpy as np
import pytest

import musim_runner as mr


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("musim")
    cfg = mr.write_config(tmp)
    return tmp, cfg, mr.build_model(tmp, cfg)


def test_append_targets_matches_golden(setup):
    _, _, model = setup
    made = (model.parent / "musculo_targets.xml").read_text()
    assert made == (mr.GOLDEN / "musculo_targets.xml").read_text()


@pytest.fixture(scope="module", params=list(mr.TRAIN_VARIANTS))
def train_run(setup, request):
    """One short training run per setup in TRAIN_VARIANTS: (variant, output folder)."""
    return request.param, mr.train(*setup, request.param)


def test_training_stats_match_golden(train_run):
    variant, folder = train_run
    golden = np.load(mr.GOLDEN / f"train_{variant}_stats.npz")
    stats = mr.load_stats(folder)
    for name in mr.STAT_NAMES:
        a, b = golden[name], stats[name]
        assert a.shape == b.shape, f"{name}: {b.shape} episodes, golden has {a.shape}"
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        if not same.all():
            ep = int(np.argmin(same))
            pytest.fail(f"{name} first differs at episode {ep}: {b[ep]!r} vs golden {a[ep]!r}")


@pytest.mark.parametrize("ckpt", ["agent_networks", "agent_networks_best"])
def test_trained_weights_match_golden(train_run, ckpt):
    variant, folder = train_run
    golden = json.loads((mr.GOLDEN / f"train_{variant}_checkpoints.json").read_text())[ckpt]
    assert mr.checkpoint_summary(folder / "checkpoint" / f"{ckpt}.pth") == golden


@pytest.fixture(scope="module")
def pretrained(setup):
    return mr.summarize_test(mr.run_pretrained(*setup))


def test_pretrained_tracking_error_matches_golden(pretrained):
    golden = json.loads((mr.GOLDEN / "pretrained_test.json").read_text())
    assert pretrained["mean_error_cm"] == golden["mean_error_cm"]


def test_pretrained_outputs_match_golden(pretrained):
    golden = json.loads((mr.GOLDEN / "pretrained_test.json").read_text())
    for name in mr.TEST_ARRAYS:
        assert pretrained["hashes"][name] == golden["hashes"][name], f"{name} changed"
