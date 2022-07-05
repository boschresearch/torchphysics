import pytest
import torch

from torchphysics.models.activation_fn import (AdaptiveActivationFunction, 
                                               ReLUn, Sinus)


def test_create_adaptive_with_tanh():
    adap_fn = AdaptiveActivationFunction(torch.nn.Tanh())
    assert isinstance(adap_fn.activation_fn, torch.nn.Tanh)
    assert adap_fn.a == 1.0
    assert adap_fn.scaling == 1.0


def test_create_adaptive_with_ReLu():
    adap_fn = AdaptiveActivationFunction(torch.nn.ReLU(), inital_a=5.0, scaling=10.0)
    assert isinstance(adap_fn.activation_fn, torch.nn.ReLU)
    assert adap_fn.a == 5.0
    assert adap_fn.a.requires_grad
    assert adap_fn.scaling == 10.0


def test_forward_of_adaptive_activation():
    input_x = torch.tensor([[1.0], [2.0], [-5.0]])
    adap_fn = AdaptiveActivationFunction(torch.nn.ReLU(), inital_a=5.0, scaling=10.0)
    output_x = adap_fn(input_x)
    assert len(output_x) == 3
    assert output_x[0] == 50.0
    assert output_x[1] == 100.0
    assert output_x[2] == 0.0


def test_relu_n():
    relun = ReLUn(10)
    assert relun.n == 10


def test_relu_n_forward():
    relun = ReLUn(2)
    in_p = torch.tensor([1, 2, -6, 4, -2])
    out = relun(in_p)
    assert relun.n == 2
    assert torch.equal(out, torch.tensor([1, 4, 0, 16, 0]))


def test_relu_n_backward():
    relun = ReLUn(2)
    in_p = torch.tensor([1.0, 2, -6, 4, -2], requires_grad=True)
    out = relun(in_p)
    deriv, = torch.autograd.grad(torch.sum(out), in_p)
    assert torch.equal(deriv, torch.tensor([2.0, 4.0, 0, 8, 0]))


def test_relu_n_backward_2():
    relun = ReLUn(3)
    in_p = torch.tensor([1.0, 2, -6, -0.1, -2], requires_grad=True)
    out = relun(in_p)
    deriv, = torch.autograd.grad(torch.sum(out), in_p)
    assert torch.equal(deriv, torch.tensor([3.0, 12.0, 0, 0, 0]))


def test_sinus_forward():
    sin = Sinus()
    in_p = torch.tensor([1, 2, -6, 4, -2])
    out = sin(in_p)
    assert torch.equal(out, torch.sin(in_p))