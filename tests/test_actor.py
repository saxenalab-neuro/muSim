"""Unit tests for the controller network (the "actor" in SAC/model.py).

The actor is an RNN: 82 inputs -> 256 hidden units -> 39 muscle commands.
These tests pin its sizes and the range and formula of the muscle commands.
"""
import numpy as np
import pytest
import torch

from SAC.model import Actor


@pytest.fixture
def actor():
    """A fresh untrained actor with the lab's sizes: 82 inputs, 39 outputs
    (muscles), 256 hidden units, RNN type. Seeded so it's the same every time."""
    torch.manual_seed(0)
    return Actor(82, 39, 256, "rnn")


# Inputs scaled by 0 (all zeros), 1 (typical) and 1000 (extreme), to show the
# commands stay in range whatever the network sees
@pytest.mark.parametrize("scale", [0.0, 1.0, 1e3])
def test_muscle_commands_between_0_and_1(actor, scale):
    """Muscle commands always stay between 0 and 1, and all sizes are as expected."""
    state = torch.randn(1, 1, 82) * scale     # [batch, time, inputs]
    h = torch.zeros(1, 1, 256)                # starting RNN memory
    with torch.no_grad():
        action, _, mean, h_next, *_ = actor.sample(state, h, sampling=True)
    assert action.shape == mean.shape == (1, 39)
    assert h_next.shape == (1, 1, 256)
    for a in (action, mean):           # sampled (training) and mean (testing)
        assert a.min() >= 0 and a.max() <= 1


def test_muscle_command_mapping(actor):
    """The exact formula from the network's raw output to a muscle command.

    Muscle command = 0.5 x tanh(raw) + 0.5, so a raw output of -50 / 0 / +50
    gives 0 / 0.5 / 1. We force the raw outputs to those values by zeroing the
    last layer's weights and setting its biases.
    """
    # Muscle command = 0.5 * tanh(mean) + 0.5, so a mean of -50 / 0 / +50 gives 0 / 0.5 / 1
    with torch.no_grad():
        actor.mean_linear.weight.zero_()
        actor.mean_linear.bias.copy_(torch.tensor([-50.0, 0.0, 50.0] * 13))   # 3 x 13 = 39 muscles
        _, _, mean, *_ = actor.sample(torch.zeros(1, 1, 82), torch.zeros(1, 1, 256), sampling=True)
    assert mean[0].tolist() == [0.0, 0.5, 1.0] * 13


def test_muscle_control_range(env):
    """The arm model's allowed muscle range is 0.002 to 1.

    This records a small mismatch: the network ignores the model's range and
    outputs 0 to 1, and MuJoCo silently raises anything below 0.002. Harmless
    for muscles today, but it would be wrong for other actuators (H5).
    """
    # The model allows 0.002 to 1. The network ignores this and outputs 0 to 1
    # (action_space=None, guide H5), and MuJoCo clamps anything below 0.002.
    np.testing.assert_array_equal(env.action_space.low, np.full(39, 0.002, dtype=np.float32))
    np.testing.assert_array_equal(env.action_space.high, np.ones(39))
    assert env.model.actuator_ctrllimited.all()     # MuJoCo enforces the range


def test_pretrained_checkpoint_fits_network():
    """The lab's trained checkpoint loads into an 82 -> 256 -> 39 actor.

    load_state_dict is strict, so it fails if any layer size differs. This
    catches a refactor that changes the network's inputs or size.
    map_location="cpu" is needed because the checkpoint was saved on a GPU (B8).
    """
    ck = torch.load("checkpoint/agent_networks_best.pth", map_location="cpu")
    Actor(82, 39, 256, "rnn").load_state_dict(ck["agent_state_dict"])   # strict: sizes must match
