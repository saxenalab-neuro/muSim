"""Tests for loading the lab's trained network on a computer without a GPU.

Why this file exists:
    checkpoint/agent_networks_best.pth was saved on a GPU machine, so its
    tensors are marked "GPU". Two things used to crash on a laptop with no GPU:
      1. SAC/sac.py: the lab config sets cuda = True, and the agent asked for
         a GPU that isn't there. It now falls back to the CPU with a warning.
      2. simulate.py: torch.load() tried to put the saved tensors back on a
         GPU. It now passes map_location, so they go on the agent's device.

These tests only run the CPU path when no GPU is present (e.g. on a Mac);
on a GPU machine the fallback test is skipped, because there is nothing to
fall back from.

Run from the repository root:
    python -m pytest tests/test_checkpoint_loading.py
"""

import os
import sys
import types
import warnings

import pytest
import torch

# Make the repository root importable, so `import simulate` finds simulate.py.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from SAC.sac import SAC_Agent  # noqa: E402  (import after the path is set up)
from simulate import Simulate  # noqa: E402

NO_GPU = not torch.cuda.is_available()

# Sizes of the lab's trained network: 82 inputs, 39 muscle commands,
# 256 recurrent units (see configs/configs.txt).
NUM_INPUTS = 82
NUM_MUSCLES = 39
HIDDEN_SIZE = 256


def make_agent(cuda):
    """Build a SAC agent with the lab's sizes and settings from configs.txt."""
    action_space = types.SimpleNamespace(shape=(NUM_MUSCLES,))
    return SAC_Agent(NUM_INPUTS, action_space, HIDDEN_SIZE,
                     lr=0.0003, gamma=0.99, tau=0.005, model="rnn",
                     multi_policy_loss=False, alpha_usim=0.1, beta_usim=0.01,
                     gamma_usim=0.001, zeta_nusim=0, cuda=cuda,
                     alpha=0.2, automatic_entropy_tuning=True)


@pytest.mark.skipif(not NO_GPU, reason="this computer has a GPU, so no fallback happens")
def test_cuda_true_without_gpu_falls_back_to_cpu():
    """cuda = True on a computer with no GPU uses the CPU and warns."""
    with pytest.warns(UserWarning, match="no GPU was found"):
        agent = make_agent(cuda=True)
    assert agent.device.type == "cpu"


def test_cuda_false_uses_cpu_without_warning():
    """cuda = False always uses the CPU, with no warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail this test
        agent = make_agent(cuda=False)
    assert agent.device.type == "cpu"


def test_lab_gpu_checkpoint_loads_on_this_computer():
    """The lab's GPU-saved network loads, and ends up on the agent's device.

    Calls Simulate.load_saved_nets_from_checkpoint with a small stand-in
    for `self`, so no MuJoCo world has to be built.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the fallback warning is tested above
        agent = make_agent(cuda=True)
    stand_in = types.SimpleNamespace(
        agent=agent,
        checkpoint_folder=os.path.join(REPO_ROOT, "checkpoint"),
        checkpoint_file="agent_networks",
    )

    Simulate.load_saved_nets_from_checkpoint(stand_in, load_best=True)

    saved = torch.load(os.path.join(REPO_ROOT, "checkpoint", "agent_networks_best.pth"),
                       map_location="cpu")["agent_state_dict"]
    for name, tensor in agent.actor.state_dict().items():
        assert tensor.device == agent.device, f"{name} is on {tensor.device}"
        assert torch.equal(tensor.cpu(), saved[name]), f"{name} was not loaded"
