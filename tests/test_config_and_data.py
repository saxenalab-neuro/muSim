"""Settings and input data, as muSim reads them today.

When settings move out of the code and into the config file, these values
must stay the same unless a change is meant to alter behaviour.
"""
import pickle

import numpy as np
import pytest

from conftest import parse_args
from SAC import kinematics_preprocessing_specs, reward_function_specs

# configs/configs.txt as parsed by main.py (minus `cuda = True`)
EXPECTED_CONFIG = {
    "model": "rnn", "hidden_size": 256, "mode": "train", "RL_algorithm": "SAC",
    "seed": 123456, "gamma": 0.99, "tau": 0.005, "lr": 0.0003, "alpha": 0.2,
    "automatic_entropy_tuning": True, "policy_batch_size": 8, "policy_replay_size": 4000,
    "multi_policy_loss": True, "batch_iters": 1, "condition_selection_strategy": "reward",
    "sim_dt": 0, "frame_repeat": 5, "n_fixedsteps": 25, "timestep_limit": 8000,
    "trajectory_scaling": [26.3157894737], "center": [[0.06, 0.083, 0.0]],
    "stimulus_feedback": False, "proprioceptive_feedback": True, "muscle_forces": False,
    "joint_feedback": False, "visual_feedback": False,
    "visual_distance_bodies": [["hand", "target0"]], "visual_velocity": [],
    "sensory_delay_timepoints": 0,
    "alpha_usim": 0.1, "beta_usim": 0.01, "gamma_usim": 0.001, "zeta_nusim": 0.0,
    "cuda": False,
}


def test_default_config_values():
    args = vars(parse_args())
    assert {k: args[k] for k in EXPECTED_CONFIG} == EXPECTED_CONFIG


def test_task_constants_hardcoded_today(env):
    # Not in the config yet. If they move there, the defaults must match these.
    assert env.threshold_user == 0.064                         # starting zone, m
    assert reward_function_specs.reward_scaling_factor == 1000
    assert reward_function_specs.threshold_crossing_penalty == -5


@pytest.fixture(scope="module")
def kinematics():
    with open("kinematics_data/kinematics.pkl", "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def neural():
    with open("nusim_neural_data/neural_activity.pkl", "rb") as f:
        return pickle.load(f)


def test_kinematics_shapes(kinematics):
    assert kinematics["marker_names"] == ["hand"]
    lengths = {s: [kinematics[s][c].shape for c in sorted(kinematics[s])] for s in ["train", "test"]}
    assert lengths == {"train": [(1, 3, 702), (1, 3, 803), (1, 3, 932), (1, 3, 1319)],
                       "test": [(1, 3, 619), (1, 3, 1106)]}


def test_kinematics_z_locked(kinematics):
    for split in ["train", "test"]:
        for kin in kinematics[split].values():
            assert np.isfinite(kin[0, :2]).all()     # x and y are tracked
            assert np.isnan(kin[0, 2]).all()         # z is locked


def test_detected_tracking_setup():
    # Derived from the data at import time
    assert kinematics_preprocessing_specs.xyz_target == [[1, 1, 0]]
    assert kinematics_preprocessing_specs.musculo_tracking == [("hand", "target0")]
    assert kinematics_preprocessing_specs.musculo_target_joints == ["box:x0", "box:y0"]


def test_neural_data_matches_kinematics(kinematics, neural):
    for split in ["train", "test"]:
        for c, na in neural[split].items():
            assert na.shape == (kinematics[split][c].shape[-1], 49)


def test_kinematics_scaled_and_centred(env, kinematics):
    for c, raw in kinematics["train"].items():
        expected = raw[0, :2] / 26.3157894737 + np.array([[0.06], [0.083]])
        np.testing.assert_array_equal(env.kin_train[c][0, :2], expected)


def test_neural_activity_normalised(env):
    for na in env.na_train.values():
        assert na.max() == 1.0
