import torch
import math

# Spike-gradient functions

# slope = 25
# """``snntorch.surrogate.slope``
# parameterizes the transition rate of the surrogate gradients."""

class CustomSurrogate(torch.autograd.Function):
    """
    Surrogate gradient of the Heaviside step function.

    **Forward pass:** Spike operator function.

        .. math::

            S=\\begin{cases} \\frac{U(t)}{U} & \\text{if U ≥ U$_{\\rm thr}$}
            \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** User-defined custom surrogate gradient function.

    The user defines the custom surrogate gradient in a separate function.
    It is passed in the forward static method and used in the backward
    static method.

    The arguments of the custom surrogate gradient function are always
    the input of the forward pass (input_), the gradient of the input 
    (grad_input) and the output of the forward pass (out).
    
    ** Important Note: The hyperparameters of the custom surrogate gradient
    function have to be defined inside of the function itself. **

    Example::

        import torch
        import torch.nn as nn
        import snntorch as snn
        from snntorch import surrogate

        def custom_fast_sigmoid(input_, grad_input, spikes):
            ## The hyperparameter slope is defined inside the function.
            slope = 25
            grad = grad_input / (slope * torch.abs(input_) + 1.0) ** 2
            return grad

        spike_grad = surrogate.custom_surrogate(custom_fast_sigmoid)

        net_seq = nn.Sequential(nn.Conv2d(1, 12, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta,
                            spike_grad=spike_grad,
                            init_hidden=True),
                    nn.Conv2d(12, 64, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta,
                            spike_grad=spike_grad,
                            init_hidden=True),
                    nn.Flatten(),
                    nn.Linear(64*4*4, 10),
                    snn.Leaky(beta=beta,
                            spike_grad=spike_grad,
                            init_hidden=True,
                            output=True)
                    ).to(device)

    """
    @staticmethod
    def forward(ctx, input_, normalized, custom_surrogate_function):
        out = (input_ > 0).float()
        ctx.save_for_backward(input_, normalized, out)
        ctx.custom_surrogate_function = custom_surrogate_function
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_, normalized, out = ctx.saved_tensors
        custom_surrogate_function = ctx.custom_surrogate_function

        grad_input = grad_output.clone()
        grad = custom_surrogate_function(input_, normalized, grad_input, out)
        return grad, None, None


def custom_surrogate(custom_surrogate_function):
    """Custom surrogate gradient enclosed within a wrapper."""
    func = custom_surrogate_function

    def inner(data, normalized):
        return CustomSurrogate.apply(data, normalized, func)

    return inner


# class InverseSpikeOperator(torch.autograd.Function):
#     """
#     Surrogate gradient of the Heaviside step function.

#     **Forward pass:** Spike operator function.

#         .. math::

#             S=\\begin{cases} \\frac{U(t)}{U} & \\text{if U ≥
#             U$_{\\rm thr}$} \\\\
#             0 & \\text{if U < U$_{\\rm thr}$}
#             \\end{cases}

#     **Backward pass:** Gradient of spike operator.

#         .. math::

#                 \\frac{∂S}{∂U}&=\\begin{cases} \\frac{1}{U}
#                 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
#                 0 & \\text{if U < U$_{\\rm thr}$}
#                 \\end{cases}

#     :math:`U_{\\rm thr}` defaults to 1, and can be modified by calling
#     ``surrogate.spike_operator(threshold=1)``.
#     .. warning:: ``threshold`` should match the threshold of the neuron,
#     which defaults to 1 as well.

#                 """

#     @staticmethod
#     def forward(ctx, input_, threshold=1):
#         out = (input_ > 0).float()
#         ctx.save_for_backward(input_, out)
#         ctx.threshold = threshold
#         return out

#     @staticmethod
#     def backward(ctx, grad_output):
#         (input_, out) = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         grad = (grad_input * out) / (input_ + ctx.threshold)
#         return grad, None


# def inverse_spike_operator(threshold=1):
#     """Spike operator gradient enclosed with a parameterized threshold."""
#     threshold = threshold

#     def inner(x):
#         return InverseSpikeOperator.apply(x, threshold)

#     return inner


# class InverseStochasticSpikeOperator(torch.autograd.Function):
#     """
#     Surrogate gradient of the Heaviside step function.

#     **Forward pass:** Spike operator function.

#         .. math::

#             S=\\begin{cases} \\frac{U(t)}{U}
#             & \\text{if U ≥ U$_{\\rm thr}$} \\\\
#             0 & \\text{if U < U$_{\\rm thr}$}
#             \\end{cases}

#     **Backward pass:** Gradient of spike operator,
#     where the subthreshold gradient is sampled from
#     uniformly distributed noise on the interval
#     :math:`(𝒰\\sim[-0.5, 0.5)+μ) σ^2`,
#     where :math:`μ` is the mean and :math:`σ^2` is the variance.

#         .. math::

#                 S&≈\\begin{cases} \\frac{U(t)}{U}
#                 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
#                 (𝒰\\sim[-0.5, 0.5) + μ) σ^2
#                 & \\text{if U < U$_{\\rm thr}$}\\end{cases} \\\\
#                 \\frac{∂S}{∂U}&=\\begin{cases} \\frac{1}{U}
#                 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
#                 (𝒰\\sim[-0.5, 0.5) + μ) σ^2
#                 & \\text{if U < U$_{\\rm thr}$}
#                 \\end{cases}

#     :math:`U_{\\rm thr}` defaults to 1, and can be modified by calling
#     ``surrogate.SSO(threshold=1)``.

#     :math:`μ` defaults to 0, and can be modified by calling
#     ``surrogate.SSO(mean=0)``.

#     :math:`σ^2` defaults to 0.2, and can be modified by calling
#     ``surrogate.SSO(variance=0.5)``.

#     The above defaults set the gradient to the following expression:

#     .. math::

#                 \\frac{∂S}{∂U}&=\\begin{cases} \\frac{1}{U}
#                 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
#                 (𝒰\\sim[-0.1, 0.1) & \\text{if U < U$_{\\rm thr}$}
#                 \\end{cases}

#     .. warning:: ``threshold`` should match the threshold of the neuron,
#     which defaults to 1 as well.

#     """

#     @staticmethod
#     def forward(ctx, input_, threshold=1, mean=0, variance=0.2):
#         out = (input_ > 0).float()
#         ctx.save_for_backward(input_, out)
#         ctx.threshold = threshold
#         ctx.mean = mean
#         ctx.variance = variance
#         return out

#     @staticmethod
#     def backward(ctx, grad_output):
#         (input_, out) = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         grad = (grad_input * out) / (input_ + ctx.threshold) + (
#             grad_input * (~out.bool()).float()
#         ) * ((torch.rand_like(input_) - 0.5 + ctx.mean) * ctx.variance)

#         return grad, None, None, None


# def ISSO(threshold=1, mean=0, variance=0.2):
#     """Stochastic spike operator gradient enclosed with a parameterized
#     threshold, mean and variance."""
#     threshold = threshold
#     mean = mean
#     variance = variance

#     def inner(x):
#         return InverseStochasticSpikeOperator.
#         apply(x, threshold, mean, variance)

#     return inner


# piecewise linear func
# tanh surrogate func