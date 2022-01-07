import torch

from torchphysics.models.parameter import Parameter
from torchphysics.problem.spaces import Points, R1, R2


def test_create_parameter():
    p = Parameter(init=1.0, space=R1('D'))
    assert isinstance(p, Points)
    assert p.requires_grad
    assert torch.equal(p.as_tensor, torch.tensor([[1.0]]))


def test_create_parameter_in_2D():
    p = Parameter(init=[1.0, 2.0], space=R2('D'))
    assert isinstance(p, Points)
    assert p.requires_grad
    assert torch.equal(p.as_tensor, torch.tensor([[1.0, 2.0]]))