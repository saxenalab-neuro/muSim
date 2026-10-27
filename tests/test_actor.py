"""The controller network: input/output sizes and the muscle command range."""
import numpy as np
import pytest
import torch

from SAC.model import Actor


@pytest.fixture
def actor():
    torch.manual_seed(0)
    return Actor(82, 39, 256, "rnn")


@pytest.mark.parametrize("scale", [0.0, 1.0, 1e3])
def test_muscle_commands_between_0_and_1(actor, scale):
    state = torch.randn(1, 1, 82) * scale
    h = torch.zeros(1, 1, 256)
    with torch.no_grad():
        action, _, mean, h_next, *_ = actor.sample(state, h, sampling=True)
    assert action.shape == mean.shape == (1, 39)
    assert h_next.shape == (1, 1, 256)
    for a in (action, mean):           # sampled (training) and mean (testing)
        assert a.min() >= 0 and a.max() <= 1


def test_muscle_command_mapping(actor):
    # Muscle command = 0.5 * tanh(mean) + 0.5, so a mean of -50 / 0 / +50 gives 0 / 0.5 / 1
    with torch.no_grad():
        actor.mean_linear.weight.zero_()
        actor.mean_linear.bias.copy_(torch.tensor([-50.0, 0.0, 50.0] * 13))
        _, _, mean, *_ = actor.sample(torch.zeros(1, 1, 82), torch.zeros(1, 1, 256), sampling=True)
    assert mean[0].tolist() == [0.0, 0.5, 1.0] * 13


def test_muscle_control_range(env):
    # The model allows 0.002 to 1. The network ignores this and outputs 0 to 1
    # (action_space=None, guide H5), and MuJoCo clamps anything below 0.002.
    np.testing.assert_array_equal(env.action_space.low, np.full(39, 0.002, dtype=np.float32))
    np.testing.assert_array_equal(env.action_space.high, np.ones(39))
    assert env.model.actuator_ctrllimited.all()


def test_pretrained_checkpoint_fits_network():
    ck = torch.load("checkpoint/agent_networks_best.pth", map_location="cpu")
    Actor(82, 39, 256, "rnn").load_state_dict(ck["agent_state_dict"])   # strict: sizes must match
