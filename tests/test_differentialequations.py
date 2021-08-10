import torch
import pytest
import numpy as np
from torchphysics.utils.differentialequations import \
    (HeatEquation, BurgersEquation, IncompressibleNavierStokesEquation)

# Test HeatEquation
def test_create_heat_equation():
    heat_eq = HeatEquation(diffusivity=30, 
                           spatial_var='y', 
                           time_var='T')
    assert heat_eq.diffusivity == 30
    assert heat_eq.spatial_var == 'y'
    assert heat_eq.time_var == 'T'


def test_heat_equation_forward():
    heat_eq = HeatEquation(diffusivity=2, 
                           spatial_var='x', 
                           time_var='t')
    x = torch.tensor([[2.0], [1.0]], requires_grad=True)
    t = torch.tensor([[0.0], [1.0]], requires_grad=True)
    input_dic = {'x': x, 't': t}
    def test_func(x, t):
        out = x**2 + x*t
        return out
    output = test_func(x, t)
    out_heat = heat_eq(output, **input_dic)
    assert out_heat.shape == (2, 1) 
    heat_np = out_heat.detach().numpy()
    assert np.allclose(heat_np, [[-2], [-3]])


# Test BurgersEquation
def test_create_burgers_equation():
    burger_eq = BurgersEquation(viscosity=3.4, 
                                spatial_var='y', 
                                time_var='T')
    assert burger_eq.viscosity == 3.4
    assert burger_eq.spatial_var == 'y'
    assert burger_eq.time_var == 'T'


def test_burgers_equation_forward():
    burger_eq = BurgersEquation(viscosity=2, 
                                spatial_var='x', 
                                time_var='t')
    x = torch.tensor([[2.0, 5.0], [1.0, 1.0]], requires_grad=True)
    t = torch.tensor([[0.0], [1.0]], requires_grad=True)
    input_dic = {'x': x, 't': t}
    def test_func(x, t):
        out = torch.zeros_like(x)
        out[:, :1] = x[:, :1]**2 + x[:, 1:]*t
        out[:, 1:] = t**2 + x[:, :1]
        return out
    output = test_func(x, t)
    out = burger_eq(output, **input_dic)
    assert out.shape == (2, 2) 
    out_np = out.detach().numpy()
    assert np.allclose(out_np, [[17, 4], [3, 4]])


def test_burgers_equation_forward_with_0_viscosity():
    burger_eq = BurgersEquation(viscosity=0, 
                                spatial_var='x', 
                                time_var='t')
    x = torch.tensor([[2.0, 5.0], [1.0, 1.0]], requires_grad=True)
    t = torch.tensor([[0.0], [1.0]], requires_grad=True)
    input_dic = {'x': x, 't': t}
    def test_func(x, t):
        out = torch.zeros_like(x)
        out[:, :1] = x[:, :1]**2 + x[:, 1:]*t
        out[:, 1:] = t**2 + x[:, :1]
        return out
    output = test_func(x, t)
    out = burger_eq(output, **input_dic)
    assert out.shape == (2, 2) 
    out_np = out.detach().numpy()
    assert np.allclose(out_np, [[21, 4], [7, 4]])


# Test IncompressibleNavierStokesEquation
def test_create_incom_navier_stokes_equation():
    with pytest.raises(NotImplementedError):
        _ = IncompressibleNavierStokesEquation(viscosity=20)