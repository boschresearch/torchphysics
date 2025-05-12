import pytest
import torch

from torchphysics.problem.domains.domainND.hypercube import HyperCube
from torchphysics.problem.spaces.space import Rn
from torchphysics.problem.spaces.points import Points


def test_create_hypercube():
    H = HyperCube(Rn("x", 4), 0.0, 10.0)
    assert "x" in H.space
    assert H.dim == 4
    assert torch.all(H.lower_bounds == torch.tensor([0.0, 0.0, 0.0, 0.0]))
    assert torch.all(H.upper_bounds == torch.tensor([10.0, 10.0, 10.0, 10.0]))


def test_create_hypercube_with_list():
    H = HyperCube(Rn("x", 2), (1.0, 2.0), 20.0)
    assert "x" in H.space
    assert H.dim == 2
    assert torch.all(H.lower_bounds == torch.tensor([1.0, 2.0]))
    assert torch.all(H.upper_bounds == torch.tensor([20.0, 20.0]))


def test_create_hypercube_with_tensor():
    H = HyperCube(Rn("x", 2), (1.0, 2.0), torch.tensor([20.0, 2.5]))
    assert "x" in H.space
    assert H.dim == 2
    assert torch.all(H.lower_bounds == torch.tensor([1.0, 2.0]))
    assert torch.all(H.upper_bounds == torch.tensor([20.0, 2.5]))


def test_create_hypercube_with_wrong_shapes():
    with pytest.raises(AssertionError):
        _ = HyperCube(Rn("x", 4), (1.0, 2.0, 3.0), torch.tensor([20.0, 2.5, 5.0, 5.0, 5.0]))
    with pytest.raises(AssertionError):
        _ = HyperCube(Rn("x", 4), (1.0, 2.0, 3.0, 0.0), torch.tensor([20.0, 2.5, 5.0, 5.0, 5.0]))


def test_create_hypercube_with_callable():
    with pytest.raises(ValueError):
        _ = HyperCube(Rn("x", 4), lambda x: x, 0.0)


def test_get_hypercube_volume():
    H = HyperCube(Rn("x", 2), (1.0, 2.0), 20.0)
    assert H.volume() == 19*18


def test_hypercube_sample_random_uniform_with_n():
    lower_bound = (1.0, 2.0, 3.0, 4.0)
    upper_bound = (10.0, 11.0, 10.0, 20.0)
    H = HyperCube(Rn("x", 4), lower_bound, upper_bound)
    points = H.sample_random_uniform(100).as_tensor
    assert points.shape == (100, 4)
    for i in range(4):
            assert all(points[:, i] >= lower_bound[i])
            assert all(points[:, i] <= upper_bound[i])


def test_hypercube_sample_random_uniform_with_d():
    lower_bound = (1.0, 2.0, 3.0, 4.0)
    upper_bound = (10.0, 11.0, 10.0, 20.0)
    H = HyperCube(Rn("x", 4), lower_bound, upper_bound)
    points = H.sample_random_uniform(d=1).as_tensor
    for i in range(4):
            assert all(points[:, i] >= lower_bound[i])
            assert all(points[:, i] <= upper_bound[i])


def test_hypercube_sample_grid_with_n():
    lower_bound = (1.0, 2.0, 3.0, 4.0)
    upper_bound = (10.0, 11.0, 10.0, 20.0)
    H = HyperCube(Rn("x", 4), lower_bound, upper_bound)
    points = H.sample_grid(100).as_tensor
    assert points.shape == (100, 4)
    for i in range(4):
            assert all(points[:, i] >= lower_bound[i])
            assert all(points[:, i] <= upper_bound[i])


def test_hypercube_sample_grid_with_d():
    lower_bound = (1.0, 2.0, 3.0, 4.0)
    upper_bound = (10.0, 11.0, 10.0, 20.0)
    H = HyperCube(Rn("x", 4), lower_bound, upper_bound)
    points = H.sample_grid(d=1).as_tensor
    for i in range(4):
            assert all(points[:, i] >= lower_bound[i])
            assert all(points[:, i] <= upper_bound[i])