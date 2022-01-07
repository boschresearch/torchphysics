import torch

from ..problem.spaces import Points


class Parameter(Points):
    """A parameter that is part of the problem and can be learned during training.
    
    Parameters
    ----------
    init : number, list, array or tensor
        The inital guess for the parameter.
    space : torchphysics.problem.spaces.Space
        The Space to which this parameter belongs. Essentially defines the 
        shape of the parameter, e.g for a single number use R1.

    Notes
    -----
    To use these Parameters during training they have to passed on to the used
    condition. If many different parameters are used they have to be connected over
    .join(), see the Points-Class for the exact usage.

    If the domains itself should depend on some parameters or the solution sholud be 
    learned for different parameter values, this class should NOT be used.
    These parameters are mostly meant for inverse problems.
    Instead, the parameters have to be defined with their own domain and samplers. 
    """
    def __init__(self, init, space, **kwargs):
        init = torch.as_tensor(init).float().reshape(1, -1)
        data = torch.nn.Parameter(init)
        super().__init__(data, space, **kwargs)
