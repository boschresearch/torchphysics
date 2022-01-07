import torch

from torchphysics.models.fcn import FCN
from torchphysics.problem.spaces import Points, R1, R2


def test_create_fcn():
    fcn = FCN(input_space=R2('x'), output_space=R1('u'), 
              hidden=(10, 10, 10))
    assert isinstance(fcn.sequential, torch.nn.Sequential)
    for i in range(0, 6, 2):
        assert isinstance(fcn.sequential[i], torch.nn.Linear)
    for i in range(1, 6, 2):
        assert isinstance(fcn.sequential[i], torch.nn.Tanh)
    assert fcn.sequential[0].in_features == 2
    assert fcn.sequential[0].out_features == 10
    for i in range(2, 4, 2):
        assert fcn.sequential[2*i].in_features == 10
        assert fcn.sequential[2*i].out_features == 10
    assert fcn.sequential[6].in_features == 10
    assert fcn.sequential[6].out_features == 1


def test_create_fcn_with_different_activation():
    fcn = FCN(input_space=R1('x'), output_space=R2('u'), 
              hidden=(10, 10, 10), 
              activations=(torch.nn.Tanh(), torch.nn.ReLU(), torch.nn.Sigmoid()))
    assert isinstance(fcn.sequential, torch.nn.Sequential)
    for i in range(0, 6, 2):
        assert isinstance(fcn.sequential[i], torch.nn.Linear)
    assert isinstance(fcn.sequential[1], torch.nn.Tanh)
    assert isinstance(fcn.sequential[3], torch.nn.ReLU)
    assert isinstance(fcn.sequential[5], torch.nn.Sigmoid)
    assert fcn.sequential[0].in_features == 1
    assert fcn.sequential[0].out_features == 10
    for i in range(2, 4, 2):
        assert fcn.sequential[2*i].in_features == 10
        assert fcn.sequential[2*i].out_features == 10
    assert fcn.sequential[6].in_features == 10
    assert fcn.sequential[6].out_features == 2


def test_create_fcn_with_different_gains():
    fcn = FCN(input_space=R2('x'), output_space=R1('u'), 
              hidden=(10, 10, 10), xavier_gains=(0.2, 0.1, 3.4))
    assert isinstance(fcn.sequential, torch.nn.Sequential)
    for i in range(0, 6, 2):
        assert isinstance(fcn.sequential[i], torch.nn.Linear)
    for i in range(1, 6, 2):
        assert isinstance(fcn.sequential[i], torch.nn.Tanh)
    assert fcn.sequential[0].in_features == 2
    assert fcn.sequential[0].out_features == 10
    for i in range(2, 4, 2):
        assert fcn.sequential[2*i].in_features == 10
        assert fcn.sequential[2*i].out_features == 10
    assert fcn.sequential[6].in_features == 10
    assert fcn.sequential[6].out_features == 1


def test_fcn_forward():
    fcn = FCN(input_space=R2('x'), output_space=R1('u'), 
              hidden=(10, 10, 10))
    test_data = Points(torch.tensor([[2, 3.0], [0, 1]]), R2('x'))
    out = fcn(test_data)
    assert isinstance(out, Points)
    assert len(out) == 2