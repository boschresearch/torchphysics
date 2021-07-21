import torch

from .differentialoperators import (laplacian,
                                    jac,
                                    grad,
                                    div)


class HeatEquation(torch.nn.Module):
    """
    Implementation of the homogenous heat equation.

    Parameters
    ----------
    diffusivity : scalar or Parameter
        The diffusivity coefficient of the heat eqaution. Should be either
        a scalar constant, or a Parameter that will be learned based on data.
    spatial_var : str
        Name of the spatial Variable. Defaults to 'x'.
    time_var : str
        Name of the time Variable. Defaults to 't'.
    """
    def __init__(self,
                 diffusivity,
                 spatial_var='x',
                 time_var='t'):
        super().__init__()
        self.diffusivity = diffusivity
        self.spatial_var = spatial_var
        self.time_var = time_var

    def forward(self, u, input):
        return grad(u, input[self.time_var]) - \
            self.diffusivity*laplacian(u, input[self.spatial_var])
