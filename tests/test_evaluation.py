import torch
import numpy as np

from torchphysics.utils.evaluation import compute_min_and_max
from torchphysics.problem.samplers import GridSampler
from torchphysics.problem.domains import Circle, Interval
from torchphysics.problem.spaces import R2, R1, Points
from torchphysics.utils import grad


def test_get_min_and_max():
    def eval_fun(x):
        out = torch.sin(x.as_tensor[:, :1])
        return Points(out, R1('u'))
    sampler = GridSampler(Circle(R2('x'), [0, 0], 6), n_points=200)
    test_min, test_max = compute_min_and_max(eval_fun, sampler)
    assert np.isclose(test_min, -1, atol=0.01)
    assert np.isclose(test_max, 1, atol=0.01)


def test_get_min_and_max_of_derivative():
    def model_fun(x):
        out = x.as_tensor**2
        return Points(out, R1('u'))
    def eval_fun(u, x):
        return grad(u, x)
    sampler = GridSampler(Interval(R1('x'), 0, 1), n_points=100)
    test_min, test_max = compute_min_and_max(model_fun, sampler, eval_fun,
                                             requieres_grad=True)
    assert torch.isclose(test_min, torch.tensor(0.0), atol=0.05)
    assert torch.isclose(test_max, torch.tensor(2.0), atol=0.05)