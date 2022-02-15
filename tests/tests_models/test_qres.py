import torch

from torchphysics.models.qres import QRES, Quadratic
from torchphysics.problem.spaces import Points, R1, R2


def test_create_quadratic_layer():
    quad = Quadratic(in_features=3, out_features=2, xavier_gains=1)
    assert isinstance(quad, torch.nn.Module)
    assert quad.bias.shape == (1, 2)
    assert quad.linear_weights.weight.shape == (2, 3)
    assert quad.quadratic_weights.weight.shape == (2, 3)


def test_create_qres():
    qres = QRES(input_space=R2('x'), output_space=R1('u'), 
                hidden=(10, 10, 10))
    assert isinstance(qres.sequential, torch.nn.Sequential)
    for i in range(0, 6, 2):
        assert isinstance(qres.sequential[i], Quadratic)
    for i in range(1, 6, 2):
        assert isinstance(qres.sequential[i], torch.nn.Tanh)
    assert qres.sequential[0].in_features == 2
    assert qres.sequential[0].out_features == 10
    for i in range(2, 4, 2):
        assert qres.sequential[2*i].in_features == 10
        assert qres.sequential[2*i].out_features == 10
    assert qres.sequential[6].in_features == 10
    assert qres.sequential[6].out_features == 1


def test_create_qres_with_different_activation():
    qres = QRES(input_space=R1('x'), output_space=R2('u'), 
                hidden=(10, 10, 10), 
                activations=(torch.nn.Tanh(), torch.nn.ReLU(), torch.nn.Sigmoid()))
    assert isinstance(qres.sequential, torch.nn.Sequential)
    for i in range(0, 6, 2):
        assert isinstance(qres.sequential[i], Quadratic)
    assert isinstance(qres.sequential[1], torch.nn.Tanh)
    assert isinstance(qres.sequential[3], torch.nn.ReLU)
    assert isinstance(qres.sequential[5], torch.nn.Sigmoid)
    assert qres.sequential[0].in_features == 1
    assert qres.sequential[0].out_features == 10
    for i in range(2, 4, 2):
        assert qres.sequential[2*i].in_features == 10
        assert qres.sequential[2*i].out_features == 10
    assert qres.sequential[6].in_features == 10
    assert qres.sequential[6].out_features == 2


def test_create_qres_with_different_gains():
    qres = QRES(input_space=R2('x'), output_space=R1('u'), 
                hidden=(10, 10, 10), xavier_gains=(0.2, 0.1, 3.4))
    assert isinstance(qres.sequential, torch.nn.Sequential)
    for i in range(0, 6, 2):
        assert isinstance(qres.sequential[i], Quadratic)
    for i in range(1, 6, 2):
        assert isinstance(qres.sequential[i], torch.nn.Tanh)
    assert qres.sequential[0].in_features == 2
    assert qres.sequential[0].out_features == 10
    for i in range(2, 4, 2):
        assert qres.sequential[2*i].in_features == 10
        assert qres.sequential[2*i].out_features == 10
    assert qres.sequential[6].in_features == 10
    assert qres.sequential[6].out_features == 1


def test_qres_forward():
    qres = QRES(input_space=R2('x'), output_space=R1('u'), 
                hidden=(10, 10, 10))
    test_data = Points(torch.tensor([[2, 3.0], [0, 1]]), R2('x'))
    out = qres(test_data)
    assert isinstance(out, Points)
    assert out.as_tensor.shape == (2, 1)