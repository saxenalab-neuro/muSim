"""Tests for the episode-end check, Muscle_Env.is_done().

Why this file exists:
    is_done() ends an episode when the tracked body leaves its zone around the
    target. It used to have the names "hand" and "target0" typed in, so a model
    whose part was called something else (e.g. a mouse "paw") crashed with
    "Invalid name 'hand'". It now reads the body/target pairs from the data
    (kinematics_preprocessing_specs.musculo_tracking, built from marker_names),
    the same pairs the reward uses.

These tests call is_done() with a small stand-in for the environment, so no
MuJoCo model has to be built: positions are set by hand, in metres.

Run from the repository root:
    python -m pytest tests/test_episode_end.py
"""

import os
import sys
import types

import numpy as np
import pytest

# Make the repository root importable, so `import SAC` works.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from SAC import kinematics_preprocessing_specs  # noqa: E402
from SAC.RL_Framework_Mujoco import Muscle_Env  # noqa: E402

ZONE = 0.008  # 0.8 cm, the smallest zone used in training


def stand_in_env(positions, istep=200, timestep_limit=1000):
    """A minimal object with what is_done() reads: the model's body ids,
    the body positions (data.xpos), the step count and the zone size."""
    names = list(positions)
    model = types.SimpleNamespace(body=lambda name: types.SimpleNamespace(id=names.index(name)))
    data = types.SimpleNamespace(xpos=np.array([positions[n] for n in names], dtype=float))
    return types.SimpleNamespace(model=model, data=data, istep=istep,
                                 timestep_limit=timestep_limit, threshold=ZONE)


@pytest.fixture
def tracking(monkeypatch):
    """Set which body tracks which target, as the data would."""
    def set_pairs(pairs):
        monkeypatch.setattr(kinematics_preprocessing_specs, "musculo_tracking", pairs)
    return set_pairs


def test_body_with_another_name_does_not_crash(tracking):
    """A part called "paw" (no "hand" anywhere) works. Inside the zone: keep going."""
    tracking([("paw", "target0")])
    env = stand_in_env({"paw": [0.1, 0.2, 0.3], "target0": [0.1, 0.2, 0.3 + 0.005]})
    assert Muscle_Env.is_done(env) is False


def test_body_leaving_its_zone_ends_the_episode(tracking):
    """"paw" 2 cm away from its target, outside the 0.8 cm zone: the episode ends."""
    tracking([("paw", "target0")])
    env = stand_in_env({"paw": [0.1, 0.2, 0.3], "target0": [0.1 + 0.02, 0.2, 0.3]})
    assert Muscle_Env.is_done(env) is True


def test_any_tracked_body_leaving_ends_the_episode(tracking):
    """With two markers, the second one leaving its zone also ends the episode.

    This matches the reward, which gives the penalty if ANY body is outside.
    """
    tracking([("hand", "target0"), ("elbow", "target1")])
    env = stand_in_env({
        "hand": [0.1, 0.2, 0.3], "target0": [0.1, 0.2, 0.3],      # on target
        "elbow": [0.0, 0.0, 0.0], "target1": [0.0, 0.05, 0.0],    # 5 cm away
    })
    assert Muscle_Env.is_done(env) is True


def test_time_limit_still_ends_the_episode(tracking):
    """On target, but at the step limit: the episode ends, as before."""
    tracking([("hand", "target0")])
    env = stand_in_env({"hand": [0.1, 0.2, 0.3], "target0": [0.1, 0.2, 0.3]},
                       istep=1000, timestep_limit=1000)
    assert Muscle_Env.is_done(env) is True


def test_lab_data_still_tracks_the_hand():
    """The lab's cycling data still pairs "hand" with "target0"."""
    assert list(kinematics_preprocessing_specs.musculo_tracking) == [("hand", "target0")]
