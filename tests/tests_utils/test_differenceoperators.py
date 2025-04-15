import pytest
import torch
import numpy as np
from torchphysics.utils.differentialoperators.differenceoperators import \
    (discrete_grad_on_grid, discrete_laplacian_on_grid)


def test_discrete_grad_in_1d():
    A = torch.ones(5, 10,1)
    A_grad = discrete_grad_on_grid(A, 1)
    assert A_grad.shape == (5, 10, 1)


def test_discrete_grad_in_2d():
    A = torch.ones(5,10,10,1)
    A_grad = discrete_grad_on_grid(A, 1)
    assert A_grad.shape == (5, 10, 10, 2)


def test_discrete_grad_in_3d():
    A = torch.ones(5,10,10,10,1)
    A_grad = discrete_grad_on_grid(A, 1)
    assert A_grad.shape == (5, 10, 10, 10, 3)


def test_discrete_grad_in_1d_multiple_outputs():
    A = torch.ones(5, 10, 2)
    A_grad = discrete_grad_on_grid(A, 1)
    assert A_grad.shape == (5, 10, 2)


def test_discrete_grad_in_2d_multiple_outputs():
    A = torch.ones(5,10,10,3)
    A_grad = discrete_grad_on_grid(A, 1)
    assert A_grad.shape == (5, 10, 10, 6)


def test_discrete_grad_in_3d_multiple_outputs():
    A = torch.ones(5,10,10,10,4)
    A_grad = discrete_grad_on_grid(A, 1)
    assert A_grad.shape == (5, 10, 10, 10, 12)


def test_discrete_grad_for_constant():
    A = torch.ones(5, 10, 10, 1)
    A_grad = discrete_grad_on_grid(A, 1)
    assert torch.allclose(A_grad, torch.zeros(5, 10, 10, 1))


def test_discrete_grad_for_linear_fn():
    A = torch.linspace(0, 1, 10).reshape(1, 10, 1).repeat(10, 1, 1)
    A_grad = discrete_grad_on_grid(A, 1.0/9.0)
    assert torch.allclose(A_grad, torch.ones(10, 10, 1))


def test_discrete_grad_for_linear_fn_scaled():
    A = torch.linspace(0, 1, 10).reshape(1, 10, 1).repeat(10, 1, 1)
    A *= 3.0
    A += 2.0
    A_grad = discrete_grad_on_grid(A, 1.0/9.0)
    assert torch.allclose(A_grad, 3.0*torch.ones(10, 10, 1))


def test_discrete_grad_for_non_linear_fn():
    X = torch.linspace(0, 1, 10).reshape(1, 10, 1).repeat(10, 1, 1)
    A = X**2
    A_grad = discrete_grad_on_grid(A, 1.0/9.0)
    assert torch.allclose(A_grad, 2*X, atol=0.001)


def test_discrete_grad_for_linear_fn_2d():
    A = torch.linspace(0, 1, 20)
    A = torch.transpose(torch.stack(torch.meshgrid((A, A))), 0, -1)
    A = A[..., :1] + A[..., 1:]
    A = A.reshape(1, 20, 20, 1).repeat(4, 1, 1, 1)
    A_grad = discrete_grad_on_grid(A, 1.0/19.0)
    assert torch.allclose(A_grad, torch.ones(4, 20, 20, 2))


def test_discrete_grad_for_linear_fn_2d_scaled():
    A = torch.linspace(0, 1, 20)
    A = torch.transpose(torch.stack(torch.meshgrid((A, A))), 0, -1)
    A = torch.transpose(A, 0, 1)
    
    A = 2.0*A[..., :1] + 0.1*A[..., 1:]
    A = A.reshape(1, 20, 20, 1).repeat(4, 1, 1, 1)
    A_grad = discrete_grad_on_grid(A, 1.0/19.0)
    assert torch.allclose(A_grad[..., :1], 2.0*torch.ones(4, 20, 20, 1), atol=0.0001)
    assert torch.allclose(A_grad[..., 1:], 0.1*torch.ones(4, 20, 20, 1), atol=0.0001)


def test_discrete_grad_for_non_linear_fn_2d():
    X = torch.linspace(0, 1, 20)
    X = torch.transpose(torch.stack(torch.meshgrid((X, X))), 0, -1)
    X = torch.transpose(X, 0, 1)
    
    A = X[..., :1] * X[..., 1:]**2 + torch.sin(X[..., :1])
    A = A.reshape(1, 20, 20, 1).repeat(4, 1, 1, 1)
    A_grad = discrete_grad_on_grid(A, 1.0/19.0)
    real_grad_1 = X[..., 1:]**2 + torch.cos(X[..., :1])
    real_grad_2 = 2*X[..., :1]*X[..., 1:]
    assert torch.allclose(A_grad[..., :1], real_grad_1, atol=0.001)
    assert torch.allclose(A_grad[..., 1:], real_grad_2, atol=0.001)


def test_discrete_grad_for_non_rectangle():
    X = torch.linspace(0, 1, 50)
    Y = torch.linspace(0, 2, 100)
    X = torch.transpose(torch.stack(torch.meshgrid((X, Y))), 0, -1)
    X = torch.transpose(X, 0, 1)

    A = X[..., :1] * X[..., 1:]**2 + torch.sin(X[..., :1])
    A = A.reshape(1, 50, 100, 1).repeat(4, 1, 1, 1)
    A_grad = discrete_grad_on_grid(A, 1.0/49.0)

    real_grad_1 = X[..., 1:]**2 + torch.cos(X[..., :1])
    real_grad_2 = 2*X[..., :1]*X[..., 1:]
    assert torch.allclose(A_grad[..., :1], real_grad_1, atol=0.05)
    assert torch.allclose(A_grad[..., 1:], real_grad_2, atol=0.05)

######################################################################################
def test_discrete_laplace_in_1d():
    A = torch.ones(5, 10,1)
    A_lap = discrete_laplacian_on_grid(A, 1)
    assert A_lap.shape == (5, 10, 1)


def test_discrete_laplace_in_2d():
    A = torch.ones(5,10,10,1)
    A_lap = discrete_laplacian_on_grid(A, 1)
    assert A_lap.shape == (5, 10, 10, 1)


def test_discrete_laplace_in_3d():
    A = torch.ones(5,10,10,10,1)
    A_lap = discrete_laplacian_on_grid(A, 1)
    assert A_lap.shape == (5, 10, 10, 10, 1)


def test_discrete_laplace_in_1d_multiple_outputs():
    A = torch.ones(5, 10, 2)
    A_lap = discrete_laplacian_on_grid(A, 1)
    assert A_lap.shape == (5, 10, 2)


def test_discrete_laplace_in_2d_multiple_outputs():
    A = torch.ones(5,10,10,3)
    A_lap = discrete_laplacian_on_grid(A, 1)
    assert A_lap.shape == (5, 10, 10, 3)


def test_discrete_laplace_for_constant_fn():
    A = torch.ones(5, 10, 10, 1)
    A_lap = discrete_laplacian_on_grid(A, 1)
    assert torch.allclose(A_lap, torch.zeros(5, 10, 10, 1))


def test_discrete_grad_for_quadratic_fn():
    X = torch.linspace(0, 1, 10).reshape(1, 10, 1).repeat(10, 1, 1)
    A = X**2
    A_lap = discrete_laplacian_on_grid(A, 1.0/9.0)
    assert torch.allclose(A_lap, 2.0*torch.ones_like(X), atol=0.001)


def test_discrete_grad_for_general_fn():
    X = torch.linspace(0, 1, 10).reshape(1, 10, 1).repeat(10, 1, 1)
    A = torch.sin(X) + 5.0*X**2 - 1.0
    A_lap = discrete_laplacian_on_grid(A, 1.0/9.0)
    real_lap = -torch.sin(X) + 10.0
    assert torch.allclose(A_lap, real_lap, atol=0.01)


def test_discrete_laplace_for_constant_fn_2d():
    A = 3.0*torch.ones(5, 10, 10, 1)
    A_lap = discrete_laplacian_on_grid(A, 1)
    assert torch.allclose(A_lap, torch.zeros(5, 10, 10, 1))


def test_discrete_laplace_for_quadratic_fn_2d():
    X = torch.linspace(0, 1, 20)
    X = torch.transpose(torch.stack(torch.meshgrid((X, X))), 0, -1)
    X = torch.transpose(X, 0, 1)

    A = X[..., :1]**2 + X[..., 1:]**2 
    A = A.reshape(1, 20, 20, 1).repeat(4, 1, 1, 1)

    A_lap = discrete_laplacian_on_grid(A, 1/19.0)
    assert torch.allclose(A_lap, 4.0*torch.ones(4, 20, 20, 1), atol=0.001)


def test_discrete_laplace_for_general_fn_2d():
    X = torch.linspace(0, 1, 20)
    X = torch.transpose(torch.stack(torch.meshgrid((X, X))), 0, -1)
    X = torch.transpose(X, 0, 1)

    A = torch.sin(X[..., :1]) + X[..., 1:]**2 + X[..., :1]**2 * X[..., 1:] 
    A = A.reshape(1, 20, 20, 1).repeat(4, 1, 1, 1)

    A_lap = discrete_laplacian_on_grid(A, 1/19.0)
    real_lap = -torch.sin(X[..., :1]) + 2.0 + 2.0 * X[..., 1:] 
    assert torch.allclose(A_lap, real_lap, atol=0.01)