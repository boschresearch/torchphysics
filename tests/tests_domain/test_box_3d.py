import pytest
import torch

from torchphysics.problem.domains.domain3D.box import (Box, BoxBoundary)
from torchphysics.problem.spaces.space import R3, R1
from torchphysics.problem.spaces.points import Points


def origin(t):
    return torch.column_stack((t, t, torch.zeros_like(t))) 

def width(t):
    return t + 1.0

def height(t):
    return t + 2.0


def test_create_Box():
    B = Box(R3('x'), [0.0, 0, 0], 1.0, 1.0, 1.0)
    assert all(torch.isclose(torch.tensor(B.origin.fun), torch.tensor([0.0, 0, 0])))
    assert all(torch.isclose(torch.tensor(B.width.fun), torch.tensor([1.0])))
    assert all(torch.isclose(torch.tensor(B.height.fun), torch.tensor([1.0])))
    assert all(torch.isclose(torch.tensor(B.depth.fun), torch.tensor([1.0])))
    assert 'x' in B.space
    assert 3 == B.dim


def test_create_Box_with_variable_corners():
    B = Box(R3('x'), origin, width, height, width)
    assert B.origin.fun == origin
    assert B.width.fun == width
    assert B.height.fun == height
    assert B.depth.fun == width


def test_create_Box_mixed_variable_corners():
    B = Box(R3('x'), origin, width, 1.0, 2.0)
    assert B.origin.fun == origin
    assert B.width.fun == width
    assert all(torch.isclose(torch.tensor(B.height.fun), torch.tensor([1.0])))
    assert all(torch.isclose(torch.tensor(B.depth.fun), torch.tensor([2.0])))

def test_call_Box():
    B = Box(R3('x'), origin, width, 1.0, 2.0)
    B = B(t=torch.tensor(2))
    assert all(torch.isclose(torch.tensor(B.width.fun), torch.tensor([3.0])))
    assert all(torch.isclose(torch.tensor(B.height.fun), torch.tensor([1.0])))
    assert all(torch.isclose(torch.tensor(B.depth.fun), torch.tensor([2.0])))


def test_Box_volume():
    B = Box(R3('x'), [0, 0, 0], 1, 2.0, 5.5)
    assert B.volume() == 11.0


def test_Box_volume_with_variable_corners():
    P = Box(R3('x'), [0.0, 0.0, 0.0], width, height, width)
    time = Points(torch.tensor([0, 1, 2.0]).reshape(-1, 1), R1('t'))
    volume = P.volume(time)
    assert all(torch.isclose(volume, torch.tensor([[2.0], [12.0], [36.0]])))


def test_bounding_box_Box():
    P = Box(R3('x'), [0, 0.0, -2.0], 2, 1, 3)
    bounds = P.bounding_box()
    assert bounds[0] == 0
    assert bounds[1] == 2
    assert bounds[2] == 0
    assert bounds[3] == 1
    assert bounds[4] == -2.0
    assert bounds[5] == 1

def test_bounding_box_Box_variable_corners():
    P = Box(R3('x'), origin, width, height, 2.0)
    time = Points(torch.tensor([1.0, 2, 3, 4]).reshape(-1, 1), R1('t'))
    bounds = P.bounding_box(time)
    assert bounds[0] == 1.0
    assert bounds[1] == 9.0
    assert bounds[2] == 1.0
    assert bounds[3] == 10.0
    assert bounds[4] == 0.0
    assert bounds[5] == 2.0


def test_Box_contains():
    P = Box(R3('x'), [0, 0.0, 0.0], 2, 1, 3)
    points = torch.tensor([[0.1, 0.1, 1.0], [1.98, 0.1, 1.5], 
                           [0.5, 0.5, -0.1], [-1, -1, -1], [1.9, 0.8, 3.2]])
    points = Points(points, R3('x'))
    inside = P._contains(points)
    assert all(inside[:2])
    assert not any(inside[2:])


def test_Box_contains_if_origin_changes():
    P = Box(R3('x'), origin, 1, 1, 1)
    points = torch.tensor([[0.0, 0.0, 0.1], [1.1, 1.5, 0.5], [1.6, 1.6, 0.1], 
                           [0.0, -0.8, 0.5], [0.0, 0.0, 0.5], [3.0, 3.0, 0.3]])
    time = Points(torch.tensor([0, 1, 1.5, 0, 1, 1.5]).reshape(-1, 1), R1('t'))
    points = Points(points, R3('x'))
    inside = P._contains(points, time)
    assert all(inside[:3])
    assert not any(inside[3:])


def test_Box_random_sampling_with_n():
    P = Box(R3('x'), [0, 0, 0], 2, 3, 1)
    points = P.sample_random_uniform(n=40)
    assert points.as_tensor.shape == (40, 3)
    assert all(points.as_tensor[:, :1] <= 2.0)
    assert all(points.as_tensor[:, 1:2] <= 3.0)
    assert all(points.as_tensor[:, 2:3] <= 1.0)
    assert all(points.as_tensor[:, :1] >= 0.0)
    assert all(points.as_tensor[:, 1:2] >= 0.0)
    assert all(points.as_tensor[:, 2:3] >= 0.0)


def test_Box_random_sampling_with_d():
    P = Box(R3('x'), [0, 0, 0], 2, 3, 1)
    points = P.sample_random_uniform(d=11)
    assert points.as_tensor.shape == (66, 3)
    assert all(points.as_tensor[:, :1] <= 2.0)
    assert all(points.as_tensor[:, 1:2] <= 3.0)
    assert all(points.as_tensor[:, 2:3] <= 1.0)
    assert all(points.as_tensor[:, :1] >= 0.0)
    assert all(points.as_tensor[:, 1:2] >= 0.0)
    assert all(points.as_tensor[:, 2:3] >= 0.0)


def test_Box_random_sampling_with_n_and_variable_origin():
    P = Box(R3('x'), origin, 2, 2, 2)
    time = Points(torch.tensor([0, 1]).reshape(-1, 1), R1('t'))
    points = P.sample_random_uniform(n=100, params=time)
    assert points.as_tensor.shape == (200, 3)
    assert all(P._contains(points, Points(torch.repeat_interleave(time, 100, dim=0), R1('t'))))


def test_Box_random_sampling_with_n_and_all_corners_variable():
    P = Box(R3('x'), origin, width, height, 3.0)
    time = Points(torch.tensor([0, 1]).reshape(-1, 1), R1('t'))
    points = P.sample_random_uniform(n=100, params=time)
    assert points.as_tensor.shape == (200, 3)
    assert all(P._contains(points, Points(torch.repeat_interleave(time, 100, dim=0), R1('t'))))


def test_Box_grid_sampling_with_n():
    P = Box(R3('x'), [0, 0, 0], 2, 3, 1)
    points = P.sample_grid(n=40)
    assert points.as_tensor.shape == (40, 3)
    assert all(points.as_tensor[:, :1] <= 2.0)
    assert all(points.as_tensor[:, 1:2] <= 3.0)
    assert all(points.as_tensor[:, 2:3] <= 1.0)
    assert all(points.as_tensor[:, :1] >= 0.0)
    assert all(points.as_tensor[:, 1:2] >= 0.0)
    assert all(points.as_tensor[:, 2:3] >= 0.0)

def test_Box_grid_sampling_with_n_perfect():
    P = Box(R3('x'), [0, 0, 0], 1, 1, 1)
    points = P.sample_grid(n=1000)
    assert points.as_tensor.shape == (1000, 3)
    assert all(points.as_tensor[:, :1] <= 1.0)
    assert all(points.as_tensor[:, 1:2] <= 1.0)
    assert all(points.as_tensor[:, 2:3] <= 1.0)
    assert all(points.as_tensor[:, :1] >= 0.0)
    assert all(points.as_tensor[:, 1:2] >= 0.0)
    assert all(points.as_tensor[:, 2:3] >= 0.0)


def test_Box_grid_sampling_with_d():
    P = Box(R3('x'), [0, 0, 0], 2, 3, 1)
    points = P.sample_grid(d=11)
    assert points.as_tensor.shape == (48, 3)
    assert all(points.as_tensor[:, :1] <= 2.0)
    assert all(points.as_tensor[:, 1:2] <= 3.0)
    assert all(points.as_tensor[:, 2:3] <= 1.0)
    assert all(points.as_tensor[:, :1] >= 0.0)
    assert all(points.as_tensor[:, 1:2] >= 0.0)
    assert all(points.as_tensor[:, 2:3] >= 0.0)


def test_Box_grid_sampling_with_n_and_variable_origin():
    P = Box(R3('x'), origin, 2, 2, 2)
    time = Points(torch.tensor([0.0]).reshape(-1, 1), R1('t'))
    points = P.sample_grid(n=100, params=time)
    assert points.as_tensor.shape == (100, 3)
    assert all(P._contains(points, Points(torch.repeat_interleave(time, 100, dim=0), R1('t'))))


def test_Box_grid_sampling_with_n_and_all_corners_variable():
    P = Box(R3('x'), origin, width, height, 3.0)
    time = Points(torch.tensor([0.0]).reshape(-1, 1), R1('t'))
    points = P.sample_grid(n=100, params=time)
    assert points.as_tensor.shape == (100, 3)
    assert all(P._contains(points, Points(torch.repeat_interleave(time, 100, dim=0), R1('t'))))


def test_get_Box_boundary():
    P = Box(R3('x'), [0, 0, 0], 1, 1, 1)
    boundary = P.boundary
    assert boundary.domain == P
    assert isinstance(boundary, BoxBoundary) 


def test_call_Box_boundary():
    P = Box(R3('x'), origin, 1, 2, 1).boundary
    P = P(t=torch.tensor(2))
    assert all(torch.isclose(P.domain.origin.fun, torch.tensor([2, 2, 0])))
    assert all(torch.isclose(torch.tensor(P.domain.width.fun), torch.tensor([1])))
    assert all(torch.isclose(torch.tensor(P.domain.height.fun), torch.tensor([2])))


def test_Box_boundary_contains():
    P = Box(R3('x'), [0, 0, 0], 1, 1, 1).boundary
    points = torch.tensor([[0, 0, 0], [0.5, 0, 0], [1, 0, 0], [0.5, 0.3, 0.0], [0.3, 1, 0.5], 
                           [0.1, 0.1, 0.1], [-0.5, 0, 0], [0.5, 0.5, 0.5], [1.5, 1.5, 0.0], 
                           [0, 0, -1]])
    points = Points(points, R3('x'))
    inside = P._contains(points)
    assert all(inside[:5])
    assert not any(inside[5:])


def test_Box_boundary_contains_if_corners_change():
    P = Box(R3('x'), origin, 1, 1, 1).boundary
    points = torch.tensor([[0, 0, 0], [1, 0, 0], [1.5, 1.5, 0.0], [1.7, 1.1, 1.0],
                           [1.5, 1.5, 0.0], [-0.1, -5, 20.0], [0, 0, 0.0], [2.1, 1.0, 0.0]])
    time = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1]).reshape(-1, 1)
    time = Points(time, R1('t'))
    points = Points(points, R3('x'))
    inside = P._contains(points, time)
    assert all(inside[:4])
    assert not any(inside[4:])


def test_Box_boundary_volume():
    P = Box(R3('x'), [0, 2.0, 3.0], 1, 1, 1).boundary
    volume = P.volume()
    assert volume.item() == 6


def test_Box_boundary_volume_with_variable_corners():
    P = Box(R3('x'), origin, width, height, 2.0).boundary
    time = Points(torch.tensor([0, 1, 2.0]).reshape(-1, 1), R1('t'))
    volume = P.volume(time)
    assert all(torch.isclose(volume, torch.tensor([[16], [32.0], [52]])))


def test_Box_boundary_random_sampling_with_n():
    P = Box(R3('x'), [0, 2.0, 3.0], 1, 1, 1).boundary
    points = P.sample_random_uniform(n=10)
    assert points.as_tensor.shape == (10, 3)
    assert all(P._contains(points))


def test_Box_boundary_random_sampling_with_d():
    P = Box(R3('x'), [0, 2.0, 3.0], 1, 1, 1).boundary
    points = P.sample_random_uniform(d=15)
    assert all(P._contains(points))


def test_Box_boundary_random_sampling_with_n_and_variable_origin():
    P = Box(R3('x'), origin, 1, 1, 1).boundary
    time = Points(torch.tensor([[0.0], [1.0], [2.0]]), R1('t'))
    points = P.sample_random_uniform(n=10, params=time)
    assert points.as_tensor.shape == (30, 3)
    time = Points(torch.repeat_interleave(time, 10, dim=0), R1('t'))
    assert all(P._contains(points, time))


def test_Box_boundary_random_sampling_with_n_and_variable_corners():
    P = Box(R3('x'), origin, width, height, 1).boundary
    time = Points(torch.tensor([[0.0], [1.0], [2.0]]), R1('t'))
    points = P.sample_random_uniform(n=1, params=time)
    assert points.as_tensor.shape == (3, 3)
    assert all(P._contains(points, time))


def test_Box_boundary_grid_sampling_with_n():
    P = Box(R3('x'), [0, 2.0, 3.0], 1, 1, 1).boundary
    points = P.sample_grid(n=10)
    assert points.as_tensor.shape == (10, 3)
    assert all(P._contains(points))


def test_Box_boundary_grid_sampling_with_d():
    P = Box(R3('x'), [0, 2.0, 3.0], 1, 1, 1).boundary
    points = P.sample_grid(d=15)
    assert all(P._contains(points))


def test_Box_boundary_grid_sampling_with_n_and_variable_origin():
    P = Box(R3('x'), origin, 1, 1, 1).boundary
    time = Points(torch.tensor([[0.0]]), R1('t'))
    points = P.sample_grid(n=50, params=time)
    assert points.as_tensor.shape == (50, 3)
    time = Points(torch.repeat_interleave(time, 50, dim=0), R1('t'))
    assert all(P._contains(points, time))


def test_Box_boundary_grid_sampling_with_n_and_variable_corners():
    P = Box(R3('x'), [0, 2.0, 3.0], width, height, 1).boundary
    time = Points(torch.tensor([[2.0]]), R1('t'))
    points = P.sample_grid(n=100, params=time)
    assert points.as_tensor.shape == (100, 3)
    time = Points(torch.repeat_interleave(time, 100, dim=0), R1('t'))
    assert all(P._contains(points, time))


def test_Box_normals():
    P = Box(R3('x'), [0, 0.0, 0.0], 1, 1, 1).boundary
    points = torch.tensor([[0.5, 0.5, 0], [0.5, 0.05, 1.0], [0, 0.5, 0.5], [1.0, 0.2, 0.7], 
                           [0.2, 0.0, 0.2], [0.5, 1.0, 0.7]])
    points = Points(points, R3('x'))
    normals = P.normal(points)
    assert normals.shape == (6, 3)
    expected_normals = torch.tensor([[0, 0, -1.0], [0, 0, 1.0], 
                                     [-1.0, 0, 0.0], [1.0, 0, 0], 
                                     [0, -1.0, 0], [0, 1.0, 0]])
    assert torch.all(torch.isclose(expected_normals, normals))


def test_Box_normals_if_corners_change():
    P = Box(R3('x'), origin, 1, 1, 1).boundary
    points = torch.tensor([[1.0, 0.5, 0.5], [1.0, 1.5, 0.5], [0.5, 0.5, 1.0]])
    points = Points(points, R3('x'))
    time = Points(torch.tensor([0, 1, 0.0]).reshape(-1, 1), R1('t'))
    normals = P.normal(points, time)
    expected_normals = torch.tensor([[1.0, 0, 0], [-1.0, 0, 0], [0, 0, 1.0]])
    assert normals.shape == (3, 3)
    one = torch.tensor(1.0)
    assert all(torch.isclose(torch.linalg.norm(normals, dim=1), one, atol=0.0001))
    assert torch.all(torch.isclose(expected_normals, normals, atol=0.0001))