import torch

from .differentialoperators import (laplacian,
                                    jac,
                                    grad,
                                    convective)


class HeatEquation(torch.nn.Module):
    """
    Implementation of the homogenous heat equation: u_t - D*laplace(u) = 0

    Parameters
    ----------
    diffusivity : scalar or Parameter
        The diffusivity coefficient of the heat equation. Should be either
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

    def forward(self, u, **inputs):
        return grad(u, inputs[self.time_var]) - \
            self.diffusivity*laplacian(u, inputs[self.spatial_var])


class BurgersEquation(torch.nn.Module):
    """
    Implementation of the viscous Burgers equation:
        u_t + (u*grad)*u - viscosity * laplace(u) = 0 

    Parameters
    ----------
    viscosity : scalar or Parameter
        The viscosity coefficient of the burgers equation. Should be either
        a scalar constant, or a Parameter that will be learned based on data.
        If 0, the inviscid Burgers equation will be solved.
    spatial_var : str
        Name of the spatial Variable. Defaults to 'x'.
    time_var : str
        Name of the time Variable. Defaults to 't'.
    """
    def __init__(self,
                 viscosity,
                 spatial_var='x',
                 time_var='t'):
        super().__init__()
        self.viscosity = viscosity
        self.spatial_var = spatial_var
        self.time_var = time_var

    def forward(self, u, **inputs):
        jac_t = jac(u, inputs[self.time_var]).squeeze(dim=2)  # time derivative (2d)
        conv = convective(u, inputs[self.spatial_var], u)  # convection term
        if self.viscosity != 0.0:
            # put laplace in one vector
            laplace_vec = torch.cat(
                [laplacian(u[:, i], inputs[self.spatial_var]) for i in range(u.shape[1])],
                dim=1)
            return jac_t + conv - self.viscosity * laplace_vec
        else:
            return jac_t + conv


class IncompressibleNavierStokesEquation(torch.nn.Module):
    """
    Implementation of the incompressible Navier Stokes equation.
    Needs two outputs of the network: The velocity field and the pressure.

    Parameters
    ----------
    viscosity : scalar or Parameter
        The viscosity coefficient of the burgers equation. Should be either
        a scalar constant, or a Parameter that will be learned based on data.
        If 0, the inviscid Burgers equation will be solved.
    spatial_var : str
        Name of the spatial Variable. Defaults to 'x'.
    time_var : str
        Name of the time Variable. Defaults to 't'.
    """
    def __init__(self,
                 viscosity,
                 spatial_var='x',
                 time_var='t'):
        super().__init__()
        raise NotImplementedError
        #self.viscosity = viscosity
        #self.spatial_var = spatial_var
        #self.time_var = time_var

    def forward(self, u, p, **inputs):
        jac_t = jac(u, inputs[self.time_var]).squeeze(dim=2)  # time derivative (2d)
        conv = convective(u, inputs[self.spatial_var], u)  # convection term
        if self.viscosity != 0.0:
            # put laplace in one vector
            laplace_vec = torch.cat(
                [laplacian(u[:, i], inputs[self.spatial_var]) for i in range(u.shape[1])],
                dim=1)
            return jac_t + conv - self.viscosity * laplace_vec
        else:
            return jac_t + conv
