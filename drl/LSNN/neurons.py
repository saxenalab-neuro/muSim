from warnings import warn
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import SpikingNeuron

__all__ = [
    "SpikingNeuron",
    "LIF",
    "_SpikeTensor",
    "_SpikeTorchConv",
]

dtype = torch.float

class ALIF(SpikingNeuron):
    """Parent class for adaptive leaky integrate and fire neuron models."""
    """Currently not using delay, may implement in future"""

    def __init__(
        self,
        beta,
        threshold=0.01,
        spike_grad=None,
        learn_threshold=False,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        reset_mechanism="zero",
        state_quant=False,
        output=False,
        graded_spikes_factor=1.0,
        learn_graded_spikes_factor=False,
    ):
        super().__init__(
            threshold,
            spike_grad,
            learn_threshold,
            surrogate_disable,
            init_hidden,
            inhibition,
            reset_mechanism,
            state_quant,
            output,
            graded_spikes_factor,
            learn_graded_spikes_factor,
        )

        self._lif_register_buffer(
            beta,
        )
        self._reset_mechanism = reset_mechanism

        self.spike_grad = spike_grad

        if self.surrogate_disable:
            self.spike_grad = self._surrogate_bypass

    def _lif_register_buffer(
        self,
        beta,
        learn_beta=False,
    ):
        """Set variables as learnable parameters else register them in the
        buffer."""
        self._beta_buffer(beta, learn_beta=False)

    def _beta_buffer(self, beta, learn_beta=False):
        if not isinstance(beta, torch.Tensor):
            beta = torch.as_tensor(beta)  # TODO: or .tensor() if no copy
        self.register_buffer("beta", beta)

    def _snn_register_buffer(
        self,
        threshold,
        learn_threshold,
        reset_mechanism,
        graded_spikes_factor,
        learn_graded_spikes_factor,
    ):
        """Set variables as learnable parameters else register them in the
        buffer."""

        self._threshold_buffer(threshold, learn_threshold=False)
        self._graded_spikes_buffer(
            graded_spikes_factor, learn_graded_spikes_factor
        )

        # reset buffer
        try:
            # if reset_mechanism_val is loaded from .pt, override
            # reset_mechanism
            if torch.is_tensor(self.reset_mechanism_val):
                self.reset_mechanism = list(SpikingNeuron.reset_dict)[
                    self.reset_mechanism_val
                ]
        except AttributeError:
            # reset_mechanism_val has not yet been created, create it
            self._reset_mechanism_buffer(reset_mechanism)

    def _threshold_buffer(self, threshold, learn_threshold=False):
        if not isinstance(threshold, torch.Tensor):
            threshold = torch.as_tensor(threshold)
        self.register_buffer("threshold", threshold)

    def alif_fire(self, mem, thresh, b):
        """Generates spike if mem > threshold.
        Returns spk."""

        if self.state_quant:
            mem = self.state_quant(mem)

        mem_shift = mem - thresh
        spk = self.spike_grad(mem_shift, b)

        spk = spk * self.graded_spikes_factor

        return spk

    def alif_mem_reset(self, mem, b, thresh):
        """Generates detached reset signal if mem > threshold.
        Returns reset."""
        mem_shift = mem - thresh
        reset = self.spike_grad(mem_shift, b).clone().detach()

        return reset

    def alif_fire_inhibition(self, batch_size, mem, thresh):
        """Generates spike if mem > threshold, only for the largest membrane.
        All others neurons will be inhibited for that time step.
        Returns spk."""
        mem_shift = mem - thresh
        index = torch.argmax(mem_shift, dim=1)
        spk_tmp = self.spike_grad(mem_shift)

        mask_spk1 = torch.zeros_like(spk_tmp)
        mask_spk1[torch.arange(batch_size), index] = 1
        spk = spk_tmp * mask_spk1
        # reset = spk.clone().detach()

        return spk

    @staticmethod
    def init_lleaky():
        """
        Used to initialize spk and mem as an empty SpikeTensor.
        ``init_flag`` is used as an attribute in the forward pass to convert
        the hidden states to the same as the input.
        """
        spk = _SpikeTensor(init_flag=False)
        mem = _SpikeTensor(init_flag=False)
        b = _SpikeTensor(init_flag=False)

        return spk, mem, b


class _SpikeTensor(torch.Tensor):
    """Inherits from torch.Tensor with additional attributes.
    ``init_flag`` is set at the time of initialization.
    When called in the forward function of any neuron, they are parsed and
    replaced with a torch.Tensor variable.
    """

    @staticmethod
    def __new__(cls, *args, init_flag=False, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __init__(
        self,
        *args,
        init_flag=True,
    ):
        # super().__init__() # optional
        self.init_flag = init_flag


def _SpikeTorchConv(*args, input_):
    """Convert SpikeTensor to torch.Tensor of the same size as ``input_``."""

    states = []
    # if len(input_.size()) == 0:
    #     _batch_size = 1  # assume batch_size=1 if 1D input
    # else:
    #     _batch_size = input_.size(0)
    if (
        len(args) == 1 and type(args) is not tuple
    ):  # if only one hidden state, make it iterable
        args = (args,)
    for arg in args:
        arg = arg.to("cpu")
        arg = torch.Tensor(arg)  # wash away the SpikeTensor class
        arg = torch.zeros_like(input_, requires_grad=True)
        states.append(arg)
    if len(states) == 1:  # otherwise, list isn't unpacked
        return states[0]

    return states
