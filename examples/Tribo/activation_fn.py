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

