"""Unit tests for the settings and the input data, as muSim reads them today.

  - Every setting in configs/configs.txt, as main.py parses it.
  - Settings that are still hardcoded in the code (not yet in the config).
  - The shape and processing of the kinematics (hand path) and neural data.

Item 2 of the refactor moves settings from the code into the config file.
When a setting moves, its expected value here must stay the same, unless a
change is meant to alter behaviour.
"""
import pickle

import numpy as np
import pytest

from conftest import parse_args
from SAC import kinematics_preprocessing_specs, reward_function_specs

# configs/configs.txt as parsed by main.py (minus `cuda = True`, which the
# tests remove because they run on CPU)
EXPECTED_CONFIG = {
    # Network and training algorithm
    "model": "rnn", "hidden_size": 256, "mode": "train", "RL_algorithm": "SAC",
    "seed": 123456, "gamma": 0.99, "tau": 0.005, "lr": 0.0003, "alpha": 0.2,
    "automatic_entropy_tuning": True, "policy_batch_size": 8, "policy_replay_size": 4000,
    "multi_policy_loss": True, "batch_iters": 1, "condition_selection_strategy": "reward",
    # Simulation timing
    "sim_dt": 0, "frame_repeat": 5, "n_fixedsteps": 25, "timestep_limit": 8000,
    # How the recorded hand path is scaled and placed in the simulator
    "trajectory_scaling": [26.3157894737], "center": [[0.06, 0.083, 0.0]],
    # Which senses the network gets
    "stimulus_feedback": False, "proprioceptive_feedback": True, "muscle_forces": False,
    "joint_feedback": False, "visual_feedback": False,
    "visual_distance_bodies": [["hand", "target0"]], "visual_velocity": [],
    "sensory_delay_timepoints": 0,
    # Regularisation weights (uSim) and matching to recorded neurons (nuSim, off)
    "alpha_usim": 0.1, "beta_usim": 0.01, "gamma_usim": 0.001, "zeta_nusim": 0.0,
    "cuda": False,
}


def test_default_config_values():
    """Every setting in the default config parses to exactly the expected value."""
    args = vars(parse_args())
    assert {k: args[k] for k in EXPECTED_CONFIG} == EXPECTED_CONFIG


def test_task_constants_hardcoded_today(env):
    """Task settings that live in the code today, not the config.

    Item 2 moves these into the config. When it does, the config's defaults
    must equal these values.
    """
    # Not in the config yet. If they move there, the defaults must match these.
    assert env.threshold_user == 0.064                         # starting zone, m
    assert reward_function_specs.reward_scaling_factor == 1000  # how fast reward falls with distance
    assert reward_function_specs.threshold_crossing_penalty == -5  # reward for leaving the zone


@pytest.fixture(scope="module")
def kinematics():
    """The raw recorded hand path, straight from the data file."""
    with open("kinematics_data/kinematics.pkl", "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def neural():
    """The raw recorded neural activity, straight from the data file."""
    with open("nusim_neural_data/neural_activity.pkl", "rb") as f:
        return pickle.load(f)


def test_kinematics_shapes(kinematics):
    """One marker ("hand"), 4 training and 2 test conditions.

    Each condition is an array of [markers, x/y/z, time samples].
    """
    assert kinematics["marker_names"] == ["hand"]
    lengths = {s: [kinematics[s][c].shape for c in sorted(kinematics[s])] for s in ["train", "test"]}
    assert lengths == {"train": [(1, 3, 702), (1, 3, 803), (1, 3, 932), (1, 3, 1319)],
                       "test": [(1, 3, 619), (1, 3, 1106)]}


def test_kinematics_z_locked(kinematics):
    """x and y are recorded, and z is all NaN.

    NaN is how the data format marks a coordinate as "locked" (not tracked).
    """
    for split in ["train", "test"]:
        for kin in kinematics[split].values():
            assert np.isfinite(kin[0, :2]).all()     # x and y are tracked
            assert np.isnan(kin[0, 2]).all()         # z is locked


def test_detected_tracking_setup():
    """What muSim works out from the data when it loads.

    - xyz_target [[1, 1, 0]]: x and y tracked, z locked.
    - musculo_tracking: the model's "hand" body tracks "target0".
    - musculo_target_joints: the target moves on the x and y joints only.
    """
    # Derived from the data at import time
    assert kinematics_preprocessing_specs.xyz_target == [[1, 1, 0]]
    assert kinematics_preprocessing_specs.musculo_tracking == [("hand", "target0")]
    assert kinematics_preprocessing_specs.musculo_target_joints == ["box:x0", "box:y0"]


def test_neural_data_matches_kinematics(kinematics, neural):
    """Every condition has one neural sample per kinematics sample, from 49 neurons."""
    for split in ["train", "test"]:
        for c, na in neural[split].items():
            assert na.shape == (kinematics[split][c].shape[-1], 49)


def test_kinematics_scaled_and_centred(env, kinematics):
    """When loaded, the hand path is divided by trajectory_scaling (26.3...)
    and shifted by center ([0.06, 0.083]), both from the config."""
    for c, raw in kinematics["train"].items():
        expected = raw[0, :2] / 26.3157894737 + np.array([[0.06], [0.083]])
        np.testing.assert_array_equal(env.kin_train[c][0, :2], expected)


def test_neural_activity_normalised(env):
    """When loaded, each condition's neural activity is divided by its maximum, so it peaks at 1."""
    for na in env.na_train.values():
        assert na.max() == 1.0
