import pytest
import torch

from torchphysics.models.activation_fn import AdaptiveActivationFunction


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
