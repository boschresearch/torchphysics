import torch
import torch.nn as nn


class AdaptiveActivationFunction(nn.Module):
    """Implementation of the adaptive activation functions used in [1].
    Will create activations of the form: activation_fn(scaling*a * x), 
    where activation_fn is an arbitrary function, a is the additional 
    hyperparameter and scaling is an additional scaling factor.

    Parameters
    ----------
    activation_fn : torch.nn.module
        The underlying function that should be used for the activation.
    inital_a : float, optional
        The inital value for the adaptive parameter a. Changes the 'slop'
        of the underlying function. Default is 1.0
    scaling : float, optional
        An additional scaling factor, such that the 'a' only has to learn only
        small values. Will stay fixed while training. Default is 1.0

    Notes
    -----
    ..  [1] Ameya D. Jagtap, Kenji Kawaguchi and George Em Karniadakis, 
        "Adaptive activation functions accelerate convergence in deep and 
        physics-informed neural networks", 2020
    """
    def __init__(self, activation_fn, inital_a=1.0, scaling=1.0):
        super().__init__()
        self.activation_fn = activation_fn
        self.a = nn.Parameter(torch.tensor(inital_a))
        self.scaling = scaling

    def forward(self, x):
        return self.activation_fn(self.scaling*self.a*x)


class relu_n(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, n):
        ctx.save_for_backward(x)
        ctx.n = n
        return torch.nn.functional.relu(x)**n

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        n = ctx.n
        grad_input = grad_output.clone()
        slice_idx = input > 0
        grad_input[slice_idx] = grad_input[slice_idx] * n*input[slice_idx]**(n-1)
        grad_input[torch.logical_not(slice_idx)] = 0
        return grad_input, None # <- for n gradient, not needed


class ReLUn(nn.Module):
    """Implementation of a smoother version of ReLU, in the 
    form of relu(x)**n.

    Parameters
    ----------
    n : float
        The power to which the inputs should be rasied before appplying the
        rectified linear unit function. 
    """
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.relu_n = relu_n()

    def forward(self, x):
        return self.relu_n.apply(x, self.n)


class Sinus(torch.nn.Module):
    """Implementation of a sinus activation.
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(input)