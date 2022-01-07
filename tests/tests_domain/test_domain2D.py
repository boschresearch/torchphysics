import pytest
import torch

from torchphysics.problem.domains.domain2D.circle import (Circle, CircleBoundary)
from torchphysics.problem.domains.domain2D.parallelogram import (Parallelogram,
                                                                ParallelogramBoundary)
from torchphysics.problem.domains.domain2D.triangle import (Triangle, TriangleBoundary)
from torchphysics.problem.spaces.space import R2, R1
from torchphysics.problem.spaces.points import Points


def radius(t):
    return t + 1 

def center(t):
    return torch.column_stack((t, torch.zeros_like(t)))


def test_create_circle():
    C = Circle(R2('x'), [0, 0], 1)
    assert all(torch.isclose(torch.tensor(C.center.fun), torch.tensor([0, 0])))
    assert C.radius.fun == 1
    assert 'x' in C.space


def test_create_circle_with_variable_bounds():
    C = Circle(R2('x'), center, radius)
    assert C.center.fun == center
    assert C.radius.fun == radius


def test_create_circle_mixed_bounds():
    C = Circle(R2('x'), center, 2.0)
    assert C.center.fun == center
    assert C.radius.fun == 2.0
    C = Circle(R2('x'), [0, 0], radius)
    assert all(torch.isclose(torch.tensor(C.center.fun), torch.tensor([0, 0])))
    assert C.radius.fun == radius


def test_call_circle():
    C = Circle(R2('x'), [0, 0], radius)
    called_C = C(t=2)
    assert all(torch.isclose(torch.tensor(called_C.center.fun),
                             torch.tensor([0, 0])))
    assert called_C.radius.fun == 3


def test_bounding_box_circle():
    C = Circle(R2('x'), [1, 0], 4)
    bounds = C.bounding_box()
    assert bounds[0] == -3
    assert bounds[1] == 5
    assert bounds[2] == -4
    assert bounds[3] == 4


def test_bounding_box_circle_variable_params():
    C = Circle(R2('x'), center, 4)
    time = Points(torch.tensor([1, 2, 3, 4]).reshape(-1, 1), R1('t'))
    bounds = C.bounding_box(time)
    assert bounds[0] == -3
    assert bounds[1] == 8
    assert bounds[2] == -4
    assert bounds[3] == 4


def test_circle_contains():
    C = Circle(R2('x'), [0, 0], 4)
    points = Points(torch.tensor([[0.0, 0.0], [0, -2], [-0.1, -8], [4.1, 0]]), R2('x'))
    inside = C._contains(points)
    assert all(inside[:2])
    assert not any(inside[2:])


def test_circle_contains_if_radius_changes():
    C = Circle(R2('x'), [0, 0], radius)
    time = Points(torch.tensor([0, 1, 5, 0.1, 1]).reshape(-1, 1), R1('t'))
    points = Points(torch.tensor([[0.0, 0.0], [0, -2], [4.5, -0.1],
                                  [4.1, 0], [-0.1, -8]]), R2('x'))
    inside = C._contains(points, time)
    assert all(inside[:3])
    assert not any(inside[3:])


def test_circle_contains_if_both_params_changes():
    C = Circle(R2('x'), center, radius)
    points = Points(torch.tensor([[0.0, 0.0], [1, 1.5],
                                  [-0.1, -8], [4.1, 0], [-0.1, -8]]), R2('x'))
    time = Points(torch.tensor([0, 1, 5, 0.1, 1]).reshape(-1, 1), R1('t'))
    inside = C._contains(points, time)
    assert all(inside[:2])
    assert not any(inside[2:])


def test_circle_random_sampling_with_n():
    C = Circle(R2('x'), [0, 0], 4)
    points = C.sample_random_uniform(n=40)
    assert points.as_tensor.shape == (40, 2)
    assert all(torch.linalg.norm(points.as_tensor, dim=1) <= 4.0)


def test_circle_random_sampling_with_d():
    C = Circle(R2('x'), [0, 0], 4)
    points = C.sample_random_uniform(d=13)
    assert all(torch.linalg.norm(points.as_tensor, dim=1) <= 4.0)


def test_circle_random_sampling_with_n_and_variable_radius():
    C = Circle(R2('x'), [0, 0], radius)
    time = Points(torch.tensor([0, 1]).reshape(-1, 1), R1('t'))
    points = C.sample_random_uniform(n=4, params=time)
    assert points.as_tensor.shape == (8, 2)
    assert all(C._contains(points, Points(torch.repeat_interleave(time, 4, dim=0), R1('t'))))


def test_circle_random_sampling_with_n_and_variable_radius_and_center():
    C = Circle(R2('x'), center, radius)
    time = Points(torch.tensor([0, 1]).reshape(-1, 1), R1('t'))
    points = C.sample_random_uniform(n=4, params=time)
    assert points.as_tensor.shape == (8, 2)
    assert all(C._contains(points, Points(torch.repeat_interleave(time, 4, dim=0), R1('t'))))


def test_circle_grid_sampling_with_n():
    C = Circle(R2('x'), [0, 0], 4)
    points = C.sample_grid(n=40)
    assert points.as_tensor.shape == (40, 2)
    assert all(torch.linalg.norm(points.as_tensor, dim=1) <= 4.0)


def test_circle_grid_sampling_with_d():
    C = Circle(R2('x'), [0, 0], 4)
    points = C.sample_grid(d=1)
    assert all(torch.linalg.norm(points.as_tensor, dim=1) <= 4.0)


def test_circle_grid_sampling_with_n_and_variable_radius_and_center():
    C = Circle(R2('x'), center, radius)
    time = Points(torch.tensor([0, 1, 2]).reshape(-1, 1), R1('t'))
    points = C.sample_grid(n=3, params=time)
    assert points.as_tensor.shape == (9, 2)
    assert all(C._contains(points, Points(torch.repeat_interleave(time, 3, dim=0), R1('t'))))


def test_get_circle_boundary():
    C = Circle(R2('x'), [0, 0], 4)
    boundary = C.boundary
    assert isinstance(boundary, CircleBoundary)
    assert C == boundary.domain


def test_call_circle_boundary():
    C = Circle(R2('x'), [0, 0], radius).boundary
    new_C = C(t=2)
    assert isinstance(new_C, CircleBoundary)
    assert new_C.domain.radius.fun == 3
    assert new_C.domain.center.fun[0] == 0
    assert new_C.domain.center.fun[1] == 0


def test_circle_boundary_contains():
    C = Circle(R2('x'), [0, 0], 4).boundary
    points = Points(torch.tensor([[0, 4], [0, -4], [-0.1, 0.5], [-1, -5]]), R2('x'))
    inside = C._contains(points)
    assert all(inside[:2])
    assert not any(inside[2:])


def test_circle_boundary_contains_if_params_change():
    C = Circle(R2('x'), [0, 0], radius).boundary
    points = Points(torch.tensor([[0, 1], [0, -1], [-2, 0], [0, 2], 
                                  [0, 1], [-1, -5]]), R2('x'))
    time = Points(torch.tensor([0, 0, 1, 1, 1, 2.0]).reshape(-1, 1), R1('t'))
    inside = C._contains(points, time)
    assert all(inside[:4])
    assert not any(inside[4:])


def test_circle_boundary_random_sampling_with_n():
    C = Circle(R2('x'), [0, 0], 4).boundary
    points = C.sample_random_uniform(n=10)
    assert points.as_tensor.shape == (10, 2)
    assert all(torch.isclose(torch.linalg.norm(points.as_tensor, dim=1),
                             torch.tensor(4.0)))


def test_circle_boundary_random_sampling_with_d():
    C = Circle(R2('x'), [0, 0], 4).boundary
    points = C.sample_random_uniform(d=15)
    assert all(torch.isclose(torch.linalg.norm(points.as_tensor, dim=1), torch.tensor(4.0)))


def test_circle_boundary_random_sampling_with_n_and_variable_domain():
    C = Circle(R2('x'), center, radius).boundary
    time = Points(torch.tensor([0.0, 1.0]).reshape(-1, 1), R1('t'))
    points = C.sample_random_uniform(n=4, params=time)
    assert points.as_tensor.shape == (8, 2)
    assert all(C._contains(points, Points(torch.repeat_interleave(time, 4, dim=0), R1('t'))))


def test_circle_boundary_grid_sampling_with_n():
    C = Circle(R2('x'), [0, 0], 4).boundary
    points = C.sample_grid(n=30)
    assert points.as_tensor.shape == (30, 2)
    assert all(torch.isclose(torch.linalg.norm(points.as_tensor, dim=1),
                             torch.tensor(4.0)))


def test_circle_boundary_grid_sampling_with_d():
    C = Circle(R2('x'), [0, 0], 4).boundary
    points = C.sample_grid(d=13)
    assert all(torch.isclose(torch.linalg.norm(points.as_tensor, dim=1),
                             torch.tensor(4.0)))


def test_circle_boundary_grid_sampling_with_n_and_variable_domain():
    C = Circle(R2('x'), center, radius).boundary
    time = Points(torch.tensor([0.0, 1.0, 2.0, 5.5]).reshape(-1, 1), R1('t'))
    points = C.sample_grid(n=2, params=time)
    assert points.as_tensor.shape == (8, 2)
    time = Points(torch.repeat_interleave(time, 2, dim=0), R1('t'))
    assert all(C._contains(points, time))


def test_circle_normals():
    C = Circle(R2('x'), [0, 0], 4).boundary
    points = Points(torch.tensor([[-4.0, 0], [4, 0], [0, 4], [0, -4]]), R2('x'))
    normals = C.normal(points)
    assert normals.shape == (4, 2)
    assert torch.all(torch.isclose(torch.tensor([[-1.0, 0], [1, 0], [0, 1], [0, -1]]),
                                   normals))


def test_circle_normals_if_domain_changes():
    C = Circle(R2('x'), center, radius).boundary
    points = Points(torch.tensor([[1, 0], [1, 2], [2.0, -3.0]]), R2('x'))
    time = Points(torch.tensor([0, 1, 2.0]).reshape(-1, 1), R1('t'))
    normals = C.normal(points, time)
    assert normals.shape == (3, 2)
    assert torch.all(torch.isclose(torch.tensor([[1.0, 0], [0, 1], [0, -1]]), normals))


# Test Parallelogram

def origin(t):
    return torch.column_stack((t, torch.zeros_like(t))) 

def vec_1(t):
    return torch.column_stack((t + 1, torch.zeros_like(t)))

def vec_2(t):
    return torch.column_stack((t, t+1))


def test_create_parallelogram():
    P = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    assert all(torch.isclose(torch.tensor(P.origin.fun), torch.tensor([0, 0])))
    assert all(torch.isclose(torch.tensor(P.corner_1.fun), torch.tensor([1, 0])))
    assert all(torch.isclose(torch.tensor(P.corner_2.fun), torch.tensor([0, 1])))
    assert 'x' in P.space
    assert 2 == P.dim


def test_create_parallelogram_with_variable_corners():
    P = Parallelogram(R2('x'), origin, vec_1, vec_2)
    assert P.origin.fun == origin
    assert P.corner_1.fun == vec_1
    assert P.corner_2.fun == vec_2


def test_create_parallelogram_mixed_variable_corners():
    P = Parallelogram(R2('x'), origin, [5, 0], vec_2)
    assert P.origin.fun == origin
    assert all(torch.isclose(torch.tensor(P.corner_1.fun), torch.tensor([5, 0])))
    assert P.corner_2.fun == vec_2
    P = Parallelogram(R2('x'), [-1, 1], [5, 0], vec_2)
    assert all(torch.isclose(torch.tensor(P.origin.fun), torch.tensor([-1, 1])))
    assert all(torch.isclose(torch.tensor(P.corner_1.fun), torch.tensor([5, 0])))
    assert P.corner_2.fun == vec_2


def test_call_parallelogram():
    P = Parallelogram(R2('x'), origin, [5, 0], vec_2)
    P = P(t=torch.tensor(2))
    assert all(torch.isclose(P.origin.fun, torch.tensor([2, 0])))
    assert all(torch.isclose(torch.tensor(P.corner_1.fun), torch.tensor([5, 0])))
    assert all(torch.isclose(P.corner_2.fun, torch.tensor([2, 3])))


def test_parallelogram_volume():
    P = Parallelogram(R2('x'), [0, 0], [2, 0], [0, 1])
    assert P.volume() == 2


def test_parallelogram_volume_with_variable_corners():
    P = Parallelogram(R2('x'), origin, [5, 0], vec_2)
    time = Points(torch.tensor([0, 1, 2.0]).reshape(-1, 1), R1('t'))
    volume = P.volume(time)
    assert all(torch.isclose(volume, torch.tensor([[5], [8.0], [9]])))


def test_parallelogram_user_function_transform():
    P = Parallelogram(R2('x'), [0, 0], [2, 0], [0, 1])
    trans_1 = P._check_shape_of_evaluated_user_function(torch.tensor([2.0]))
    assert trans_1 == 2.0
    trans_2 = P._check_shape_of_evaluated_user_function(2.0)
    assert trans_2 == 2.0


def test_bounding_box_parallelogram():
    P = Parallelogram(R2('x'), [0, 0], [2, 0], [0, 1])
    bounds = P.bounding_box()
    assert bounds[0] == 0
    assert bounds[1] == 2
    assert bounds[2] == 0
    assert bounds[3] == 1


def test_bounding_box_parallelogram_variable_corners():
    P = Parallelogram(R2('x'), origin, [5, 0], vec_2)
    time = Points(torch.tensor([1.0, 2, 3, 4]).reshape(-1, 1), R1('t'))
    bounds = P.bounding_box(time)
    assert bounds[0] == 1
    assert bounds[1] == 5
    assert bounds[2] == 0
    assert bounds[3] == 5


def test_parallelogram_contains():
    P = Parallelogram(R2('x'), [0, 0], [2, 0], [0, 1])
    points = torch.tensor([[0.0, 0.0], [2, 0], [0.5, 0.5], [1.8, 0.7], [0.1, 0.9], 
                           [-1, -1], [-0.1, 0], [2.1, 0], [0, 1.01], [1, 1.1]])
    points = Points(points, R2('x'))
    inside = P._contains(points)
    assert all(inside[:5])
    assert not any(inside[5:])


def test_parallelogram_contains_if_origin_changes():
    P = Parallelogram(R2('x'), origin, [5, 0], [0, 1])
    points = torch.tensor([[0.0, 0.0], [1.0, 0], [1.6, 0.5], [0.2, 0.8],
                           [0.0, 0.0], [-2, 0], [0.4, 0.0]])
    time = Points(torch.tensor([0, 1, 1.5, 0.1, 1, -1, 0.5]).reshape(-1, 1), R1('t'))
    points = Points(points, R2('x'))
    inside = P._contains(points, time)
    assert all(inside[:4])
    assert not any(inside[4:])


def test_parallelogram_contains_if_all_corners_change():
    P = Parallelogram(R2('x'), origin, vec_1, vec_2)
    points = torch.tensor([[0.1, 0.0], [1, 2], [6, 6], 
                           [-23, 0], [-0.1, -8]])
    time = Points(torch.tensor([0.1, 1, 5, 0.1, 1]).reshape(-1, 1), R1('t'))
    points = Points(points, R2('x'))
    inside = P._contains(points, time)
    assert all(inside[:3])
    assert not any(inside[3:])


def test_parallelogram_random_sampling_with_n():
    P = Parallelogram(R2('x'), [0, 0], [2, 0], [0, 1])
    points = P.sample_random_uniform(n=40)
    assert points.as_tensor.shape == (40, 2)
    assert all(points.as_tensor[:, :1] <= 2.0)
    assert all(points.as_tensor[:, 1:] <= 1.0)
    assert all(points.as_tensor[:, :1] >= 0.0)
    assert all(points.as_tensor[:, 1:] >= 0.0)


def test_parallelogram_random_sampling_with_d():
    P = Parallelogram(R2('x'), [0, 0], [2, 0], [0, 1])
    points = P.sample_random_uniform(d=11)
    assert all(points.as_tensor[:, :1] <= 2.0)
    assert all(points.as_tensor[:, 1:] <= 1.0)
    assert all(points.as_tensor[:, :1] >= 0.0)
    assert all(points.as_tensor[:, 1:] >= 0.0)


def test_parallelogram_random_sampling_with_n_and_variable_origin():
    P = Parallelogram(R2('x'), origin, [5, 0], [0, 1])
    time = Points(torch.tensor([0, 1]).reshape(-1, 1), R1('t'))
    points = P.sample_random_uniform(n=100, params=time)
    assert points.as_tensor.shape == (200, 2)
    assert all(P._contains(points, Points(torch.repeat_interleave(time, 100, dim=0), R1('t'))))


def test_parallelogram_random_sampling_with_n_and_all_corners_variable():
    P = Parallelogram(R2('x'), origin, vec_1, vec_2)
    time = Points(torch.tensor([0, 1]).reshape(-1, 1), R1('t'))
    points = P.sample_random_uniform(n=100, params=time)
    assert points.as_tensor.shape == (200, 2)
    assert all(P._contains(points, Points(torch.repeat_interleave(time, 100, dim=0), R1('t'))))


def test_parallelogram_grid_sampling_with_n():
    P = Parallelogram(R2('x'), [0, 0], [2, 0], [0, 1])
    points = P.sample_grid(n=40)
    assert points.as_tensor.shape == (40, 2)
    assert all(points.as_tensor[:, :1] <= 2.0)
    assert all(points.as_tensor[:, 1:] <= 1.0)
    assert all(points.as_tensor[:, :1] >= 0.0)
    assert all(points.as_tensor[:, 1:] >= 0.0)


def test_parallelogram_grid_sampling_with_d():
    P = Parallelogram(R2('x'), [0, 0], [2, 0], [0, 1])
    points = P.sample_grid(d=11)
    assert all(points.as_tensor[:, :1] <= 2.0)
    assert all(points.as_tensor[:, 1:] <= 1.0)
    assert all(points.as_tensor[:, :1] >= 0.0)
    assert all(points.as_tensor[:, 1:] >= 0.0)


def test_parallelogram_grid_sampling_with_n_and_variable_origin():
    P = Parallelogram(R2('x'), origin, [5, 0], [0, 1])
    time = Points(torch.tensor([0.0]).reshape(-1, 1), R1('t'))
    points = P.sample_grid(n=100, params=time)
    assert points.as_tensor.shape == (100, 2)
    assert all(P._contains(points, Points(torch.repeat_interleave(time, 100, dim=0), R1('t'))))


def test_parallelogram_grid_sampling_with_n_and_all_corners_variable():
    P = Parallelogram(R2('x'), origin, vec_1, vec_2)
    time = Points(torch.tensor([0.0]).reshape(-1, 1), R1('t'))
    points = P.sample_grid(n=100, params=time)
    assert points.as_tensor.shape == (100, 2)
    assert all(P._contains(points, Points(torch.repeat_interleave(time, 100, dim=0), R1('t'))))


def test_get_parallelogram_boundary():
    P = Parallelogram(R2('x'), [0, 0], [2, 0], [0, 1])
    boundary = P.boundary
    assert boundary.domain == P
    assert isinstance(boundary, ParallelogramBoundary) 


def test_call_parallelogram_boundary():
    P = Parallelogram(R2('x'), origin, [5, 0], vec_2).boundary
    P = P(t=torch.tensor(2))
    assert all(torch.isclose(P.domain.origin.fun, torch.tensor([2, 0])))
    assert all(torch.isclose(torch.tensor(P.domain.corner_1.fun), torch.tensor([5, 0])))
    assert all(torch.isclose(P.domain.corner_2.fun, torch.tensor([2, 3])))


def test_parallelogram_boundary_contains():
    P = Parallelogram(R2('x'), [0, 0], [2, 0], [0, 1]).boundary
    points = torch.tensor([[0, 0], [0.5, 0], [2, 0], [2, 0.3], [2, 1], [1.4, 1], 
                           [0, 1], [0, 0.3], [0.5, 0.5], [-0.1, 0], [1.1, -0.4]])
    points = Points(points, R2('x'))
    inside = P._contains(points)
    assert all(inside[:8])
    assert not any(inside[8:])


def test_parallelogram_boundary_contains_if_corners_change():
    P = Parallelogram(R2('x'), origin, vec_1, vec_2).boundary
    points = torch.tensor([[0, 0], [1, 0], [1, 0], [1, 2],
                           [2, 1], [-0.1, -5], [3.2, 1], [1.1, 0.5]])
    time = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1]).reshape(-1, 1)
    time = Points(time, R1('t'))
    points = Points(points, R2('x'))
    inside = P._contains(points, time)
    assert all(inside[:4])
    assert not any(inside[4:])


def test_parallelogram_boundary_volume():
    P = Parallelogram(R2('x'), [0, 0], [2, 0], [0, 1]).boundary
    volume = P.volume()
    assert volume.item() == 6


def test_parallelogram_boundary_volume_with_variable_corners():
    P = Parallelogram(R2('x'), origin, [5, 0], vec_2).boundary
    time = Points(torch.tensor([0, 1, 2.0]).reshape(-1, 1), R1('t'))
    volume = P.volume(time)
    assert all(torch.isclose(volume, torch.tensor([[12], [12.0], [12]])))


def test_parallelogram_normals():
    P = Parallelogram(R2('x'), [0, 0], [2, 0], [0, 1]).boundary
    points = torch.tensor([[1.0, 0], [2, 0.5], [1.2, 1], [0.0, 0.1], 
                           [0, 0], [2, 0], [2, 1], [0, 1]])
    points = Points(points, R2('x'))
    normals = P.normal(points)
    assert normals.shape == (8, 2)
    root = torch.sqrt(torch.tensor(1/2.0))
    expected_normals = torch.tensor([[0, -1.0], [1, 0], [0, 1], [-1, 0], 
                                     [-root, -root], [root, -root], [root, root], 
                                     [-root, root]])
    assert torch.all(torch.isclose(expected_normals, normals))


def test_parallelogram_normals_if_corners_change():
    P = Parallelogram(R2('x'), origin, vec_1, [0, 1]).boundary
    points = torch.tensor([[0, 1], [0, 1], [2.0, 0.0],
                           [3.5, 1], [0.5, 0.5], [0.0, 0.8]])
    points = Points(points, R2('x'))
    time = Points(torch.tensor([0, 1, 2.0, 3.0, 1.0, 0.0]).reshape(-1, 1), R1('t'))
    normals = P.normal(points, time)
    expected_normals = torch.tensor([[-0.7071,  0.7071], [-0.9239,  0.3827],
                                     [-0.2298, -0.9732], [ 0.0000,  1.0000],
                                     [-0.7071, -0.7071], [-1.0000,  0.0000]])
    assert normals.shape == (6, 2)
    one = torch.tensor(1.0)
    assert all(torch.isclose(torch.linalg.norm(normals, dim=1), one, atol=0.0001))
    assert torch.all(torch.isclose(expected_normals, normals, atol=0.0001))


def test_parallelogram_boundary_random_sampling_with_n():
    P = Parallelogram(R2('x'), [0, 0], [2, 0], [0, 1]).boundary
    points = P.sample_random_uniform(n=10)
    assert points.as_tensor.shape == (10, 2)
    assert all(P._contains(points))


def test_parallelogram_boundary_random_sampling_with_d():
    P = Parallelogram(R2('x'), [0, 0], [2, 0], [0, 1]).boundary
    points = P.sample_random_uniform(d=15)
    assert all(P._contains(points))


def test_parallelogram_boundary_random_sampling_with_n_and_variable_origin():
    P = Parallelogram(R2('x'), origin, [5, 0], [0, 1]).boundary
    time = Points(torch.tensor([[0.0], [1.0], [2.0]]), R1('t'))
    points = P.sample_random_uniform(n=10, params=time)
    assert points.as_tensor.shape == (30, 2)
    time = Points(torch.repeat_interleave(time, 10, dim=0), R1('t'))
    assert all(P._contains(points, time))


def test_parallelogram_boundary_random_sampling_with_n_and_variable_corners():
    P = Parallelogram(R2('x'), origin, vec_1, vec_2).boundary
    time = Points(torch.tensor([[0.0], [1.0], [2.0]]), R1('t'))
    points = P.sample_random_uniform(n=1, params=time)
    assert points.as_tensor.shape == (3, 2)
    assert all(P._contains(points, time))


def test_parallelogram_boundary_grid_sampling_with_n():
    P = Parallelogram(R2('x'), [0, 0], [2, 0], [0, 1]).boundary
    points = P.sample_grid(n=10)
    assert points.as_tensor.shape == (10, 2)
    assert all(P._contains(points))


def test_parallelogram_boundary_grid_sampling_with_d():
    P = Parallelogram(R2('x'), [0, 0], [2, 0], [0, 1]).boundary
    points = P.sample_grid(d=15)
    assert all(P._contains(points))


def test_parallelogram_boundary_grid_sampling_with_n_and_variable_origin():
    P = Parallelogram(R2('x'), origin, [5, 0], [0, 1]).boundary
    time = Points(torch.tensor([[0.0], [1.0], [2.0]]), R1('t'))
    points = P.sample_grid(n=50, params=time)
    assert points.as_tensor.shape == (150, 2)
    time = Points(torch.repeat_interleave(time, 50, dim=0), R1('t'))
    assert all(P._contains(points, time))


def test_parallelogram_boundary_grid_sampling_with_n_and_variable_corners():
    P = Parallelogram(R2('x'), origin, vec_1, vec_2).boundary
    time = Points(torch.tensor([[0.0], [1.0], [2.0]]), R1('t'))
    points = P.sample_grid(n=100, params=time)
    assert points.as_tensor.shape == (300, 2)
    time = Points(torch.repeat_interleave(time, 100, dim=0), R1('t'))
    assert all(P._contains(points, time))






# Test Triangle

def origin(t):
    return torch.column_stack((t, torch.zeros_like(t))) 

def vec_1(t):
    return torch.column_stack((t + 1, torch.zeros_like(t)))

def vec_2(t):
    return torch.column_stack((t, t+1))


def test_create_triangle():
    T = Triangle(R2('x'), [0, 0], [1, 0], [0, 1])
    assert all(torch.isclose(torch.tensor(T.origin.fun), torch.tensor([0, 0])))
    assert all(torch.isclose(torch.tensor(T.corner_1.fun), torch.tensor([1, 0])))
    assert all(torch.isclose(torch.tensor(T.corner_2.fun), torch.tensor([0, 1])))
    assert 'x' in T.space
    assert 2 == T.dim


def test_create_triangle_with_variable_corners():
    T = Triangle(R2('x'), origin, vec_1, vec_2)
    assert T.origin.fun == origin
    assert T.corner_1.fun == vec_1
    assert T.corner_2.fun == vec_2


def test_create_triangle_mixed_variable_corners():
    T = Triangle(R2('x'), origin, [5, 0], vec_2)
    assert T.origin.fun == origin
    assert all(torch.isclose(torch.tensor(T.corner_1.fun), torch.tensor([5, 0])))
    assert T.corner_2.fun == vec_2
    T = Triangle(R2('x'), [-1, 1], [5, 0], vec_2)
    assert all(torch.isclose(torch.tensor(T.origin.fun), torch.tensor([-1, 1])))
    assert all(torch.isclose(torch.tensor(T.corner_1.fun), torch.tensor([5, 0])))
    assert T.corner_2.fun == vec_2


def test_call_triangle():
    T = Triangle(R2('x'), origin, [5, 0], vec_2)
    T = T(t=torch.tensor(2))
    assert all(torch.isclose(T.origin.fun, torch.tensor([2, 0])))
    assert all(torch.isclose(torch.tensor(T.corner_1.fun), torch.tensor([5, 0])))
    assert all(torch.isclose(T.corner_2.fun, torch.tensor([2, 3])))


def test_triangle_volume():
    T = Triangle(R2('x'), [0, 0], [2, 0], [0, 1])
    assert T.volume() == 1


def test_triangle_volume_with_variable_corners():
    T = Triangle(R2('x'), origin, [5, 0], vec_2)
    time = Points( torch.tensor([0, 1, 2.0]).reshape(-1, 1), R1('t'))
    volume = T.volume(time)
    assert all(torch.isclose(volume, torch.tensor([[2.5], [4.0], [4.5]])))


def test_triangle_user_function_transform():
    T = Triangle(R2('x'), [0, 0], [2, 0], [0, 1])
    trans_1 = T._check_shape_of_evaluated_user_function(torch.tensor([2.0]))
    assert trans_1 == 2.0
    trans_2 = T._check_shape_of_evaluated_user_function(2.0)
    assert trans_2 == 2.0


def test_bounding_box_triangle():
    T = Triangle(R2('x'), [0, 0], [2, 0], [0, 1])
    bounds = T.bounding_box()
    assert bounds[0] == 0
    assert bounds[1] == 2
    assert bounds[2] == 0
    assert bounds[3] == 1


def test_bounding_box_triangle_variable_corners():
    T = Triangle(R2('x'), origin, [5, 0], vec_2)
    time = Points(torch.tensor([1.0, 2, 3, 4]).reshape(-1, 1), R1('t'))
    bounds = T.bounding_box(time)
    assert bounds[0] == 1
    assert bounds[1] == 5
    assert bounds[2] == 0
    assert bounds[3] == 5


def test_triangle_contains():
    T = Triangle(R2('x'), [0, 0], [2, 0], [0, 1])
    points = torch.tensor([[0.0, 0.0], [1.9, 0], [0.4, 0.4], [0.1, 0.9], [1.8, 0.7],  
                           [-1, -1], [-0.1, 0], [2.1, 0], [0, 1.01], [1, 1.1]])
    points = Points(points, R2('x'))
    inside = T._contains(points)
    assert all(inside[:4])
    assert not any(inside[4:])


def test_triangle_contains_if_origin_changes():
    T = Triangle(R2('x'), origin, [5, 0], [0, 1])
    points = torch.tensor([[0.0, 0.0], [1.0, 0], [1.6, 0.5], [0.2, 0.8],
                           [0.0, 0.0], [-2, 0], [0.4, 0.0]])
    time = torch.tensor([0, 1, 1.5, 0.1, 1, -1, 0.5]).reshape(-1, 1)
    points = Points(points, R2('x'))
    time = Points(time, R1('t'))
    inside = T._contains(points, time)
    assert all(inside[:4])
    assert not any(inside[4:])


def test_triangle_contains_if_all_corners_change():
    T = Triangle(R2('x'), origin, vec_1, vec_2)
    points = torch.tensor([[0.1, 0.0], [1, 2], [6, 6], 
                           [-23, 0], [-0.1, -8]])
    time = torch.tensor([0.1, 1, 5, 0.1, 1]).reshape(-1, 1)
    points = Points(points, R2('x'))
    time = Points(time, R1('t'))
    inside = T._contains(points, time)
    assert all(inside[:2])
    assert not any(inside[2:])


def test_triangle_random_sampling_with_n():
    T = Triangle(R2('x'), [0, 0], [1, 0], [0, 1])
    points = T.sample_random_uniform(n=44)
    assert points.as_tensor.shape == (44, 2)
    assert all(points.as_tensor[:, :1] + points.as_tensor[:, 1:] <= 1.0)
    assert all(points.as_tensor[:, :1] >= 0.0)
    assert all(points.as_tensor[:, 1:] >= 0.0)


def test_triangle_random_sampling_with_d():
    T = Triangle(R2('x'), [0, 0], [1, 0], [0, 1])
    points = T.sample_random_uniform(d=11)
    assert all(points.as_tensor[:, :1] + points.as_tensor[:, 1:] <= 1.0)
    assert all(points.as_tensor[:, :1] >= 0.0)
    assert all(points.as_tensor[:, 1:] >= 0.0)


def test_triangle_random_sampling_with_n_and_variable_origin():
    T = Triangle(R2('x'), origin, [5, 0], [0, 1])
    time = Points(torch.tensor([0, 1]).reshape(-1, 1), R1('t'))
    points = T.sample_random_uniform(n=100, params=time)
    assert points.as_tensor.shape == (200, 2)
    assert all(T._contains(points, Points(torch.repeat_interleave(time, 100, dim=0), R1('t'))))


def test_triangle_random_sampling_with_n_and_all_corners_variable():
    T = Triangle(R2('x'), origin, vec_1, vec_2)
    time = Points(torch.tensor([0, 1]).reshape(-1, 1), R1('t'))
    points = T.sample_random_uniform(n=113, params=time)
    assert points.as_tensor.shape == (226, 2)
    assert all(T._contains(points, Points(torch.repeat_interleave(time, 113, dim=0), R1('t'))))


def test_triangle_grid_sampling_with_n():
    T = Triangle(R2('x'), [0, 0], [1, 0], [0, 1])
    points = T.sample_grid(n=42)
    assert points.as_tensor.shape == (42, 2)
    assert all(points.as_tensor[:, :1] + points.as_tensor[:, 1:] <= 1.0)
    assert all(points.as_tensor[:, :1] >= 0.0)
    assert all(points.as_tensor[:, 1:] >= 0.0)


def test_triangle_grid_sampling_with_d():
    T = Triangle(R2('x'), [0, 0], [1, 0], [0, 1])
    points = T.sample_grid(d=11)
    assert all(points.as_tensor[:, :1] + points.as_tensor[:, 1:] <= 1.0)
    assert all(points.as_tensor[:, :1] >= 0.0)
    assert all(points.as_tensor[:, 1:] >= 0.0)


def test_triangle_grid_sampling_with_n_and_variable_origin():
    T = Triangle(R2('x'), origin, [5, 0], [0, 1])
    time = Points(torch.tensor([[0.0]]), R1('t'))
    points = T.sample_grid(n=10, params=time)
    assert points.as_tensor.shape == (10, 2)
    assert all(T._contains(points, Points(torch.repeat_interleave(time, 10, dim=0), R1('t'))))


def test_triangle_grid_sampling_with_n_and_all_corners_variable():
    T = Triangle(R2('x'), origin, vec_1, vec_2)
    time = Points(torch.tensor([[0.0]]), R1('t'))
    points = T.sample_grid(n=100, params=time)
    assert points.as_tensor.shape == (100, 2)
    assert all(T._contains(points, Points(torch.repeat_interleave(time, 100, dim=0), R1('t'))))


def test_get_triangle_boundary():
    T = Triangle(R2('x'), [0, 0], [2, 0], [0, 1])
    boundary = T.boundary
    assert boundary.domain == T
    assert isinstance(boundary, TriangleBoundary) 


def test_call_triangle_boundary():
    T = Parallelogram(R2('x'), origin, [5, 0], vec_2).boundary
    T = T(t=torch.tensor(2))
    assert all(torch.isclose(T.domain.origin.fun, torch.tensor([2, 0])))
    assert all(torch.isclose(torch.tensor(T.domain.corner_1.fun), torch.tensor([5, 0])))
    assert all(torch.isclose(T.domain.corner_2.fun, torch.tensor([2, 3])))


def test_triangle_boundary_contains():
    T = Triangle(R2('x'), [0, 0], [1, 0], [0, 1]).boundary
    points = torch.tensor([[0, 0], [0.5, 0], [1, 0], [0.5, 0.5], [0.3, 0.7], [0, 0.2], 
                           [0, 1], [0, -0.3], [-0.5, -0.5], [-0.1, 0], [1.1, -0.4]])
    points = Points(points, R2('x'))
    inside = T._contains(points)
    assert all(inside[:7])
    assert not any(inside[7:])


def test_triangle_boundary_contains_if_corners_change():
    T = Triangle(R2('x'), origin, vec_1, vec_2).boundary
    points = torch.tensor([[0, 0], [1, 0], [1, 0], [1.5, 1],
                           [2, 1], [-0.1, -5], [3.2, 1], [1.1, 0.5]])
    time = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1]).reshape(-1, 1)
    points = Points(points, R2('x'))
    time = Points(time, R1('t'))
    inside = T._contains(points, time)
    assert all(inside[:4])
    assert not any(inside[4:])


def test_triangle_boundary_volume():
    T = Triangle(R2('x'), [0, 0], [2, 0], [0, 1]).boundary
    volume = T.volume()
    assert volume.item() == torch.sqrt(torch.tensor([5])) + 3


def test_triangle_fill_up_missing_grid():
    T = Triangle(R2('x'), [0, 0], [2, 0], [0, 1])
    points = torch.tensor([[1.0, 2.0], [2.0, 2.0]])
    points = T._grid_has_n_points(5, points, device='cpu')
    assert points.shape == (5, 2)


def test_triangle_boundary_volume_with_variable_corners():
    T = Triangle(R2('x'), origin, [5, 0], vec_2).boundary
    times = Points(torch.tensor([0, 1, 2.0]).reshape(-1, 1), R1('t'))
    volume = T.volume(times)
    expected = torch.sqrt((5-times.as_tensor)**2 + (times.as_tensor+1)**2) + 6
    assert all(torch.isclose(volume, expected))


def test_triangle_normals():
    T = Triangle(R2('x'), [0, 0], [1, 0], [0, 1]).boundary
    points = Points(torch.tensor([[0.7, 0], [0, 0.5], [0.5, 0.5],
                                  [0.3, 0.7], [0.1, 0], [0, 0]]), R2('x'))
    normals = T.normal(points)
    assert normals.shape == (6, 2)
    root = torch.sqrt(torch.tensor(1/2.0))
    expected_normals = torch.tensor([[0, -1.0], [-1, 0], [root, root], [root, root], 
                                     [0, -1.0], [-root, -root]])
    assert torch.all(torch.isclose(expected_normals, normals))


def test_triangle_normals_if_corners_change():
    T = Triangle(R2('x'), origin, vec_1, [0, 1]).boundary
    points = torch.tensor([[0, 1], [0, 1], [2.0, 0.0], [2, 0.5], 
                           [0.5, 0.5], [0.0, 0.8]])
    points = Points(points, R2('x'))
    time = Points(torch.tensor([0, 1, 2.0, 3.0, 1.0, 0.0]).reshape(-1, 1), R1('t'))
    normals = T.normal(points, time)
    expected_normals = torch.tensor([[-0.3827,  0.9239], [-0.8112,  0.5847],
                                     [-0.2298, -0.9732], [ 0.2425,  0.9701],
                                     [-0.7071, -0.7071], [-1.0000,  0.0000]])
    assert normals.shape == (6, 2)
    one = torch.tensor(1.0)
    assert all(torch.isclose(torch.linalg.norm(normals, dim=1), one, atol=0.0001))
    assert torch.all(torch.isclose(expected_normals, normals, atol=0.0001))


def test_triangle_boundary_random_sampling_with_n():
    T = Triangle(R2('x'), [0, 0], [2, 0], [0, 1]).boundary
    points = T.sample_random_uniform(n=32)
    assert points.as_tensor.shape == (32, 2)
    assert all(T._contains(points))


def test_triangle_boundary_random_sampling_with_d():
    T = Triangle(R2('x'), [0, 0], [2, -1], [1, 1]).boundary
    points = T.sample_random_uniform(d=16)
    assert all(T._contains(points))


def test_triangle_boundary_random_sampling_with_n_and_variable_origin():
    T = Triangle(R2('x'), origin, [5, 0], [0, 1]).boundary
    time = Points(torch.tensor([[0.0], [1.0], [2.0]]), R1('t'))
    points = T.sample_random_uniform(n=11, params=time)
    assert points.as_tensor.shape == (33, 2)
    times = Points(torch.repeat_interleave(time, 11, dim=0), R1('t'))
    assert all(T._contains(points, times))


def test_triangle_boundary_random_sampling_with_n_and_variable_corners():
    T = Triangle(R2('x'), origin, vec_1, vec_2).boundary
    time = Points(torch.tensor([[0.0], [1.0], [2.0]]), R1('t'))
    points = T.sample_random_uniform(n=1, params=time)
    assert points.as_tensor.shape == (3, 2)
    assert all(T._contains(points, time))


def test_triangle_boundary_grid_sampling_with_n():
    T = Triangle(R2('x'), [0, 0], [2, 0], [0, 1]).boundary
    points = T.sample_grid(n=10)
    assert points.as_tensor.shape == (10, 2)
    assert all(T._contains(points))


def test_triangle_boundary_grid_sampling_with_d():
    T = Triangle(R2('x'), [0, 0], [2, 0], [0, 1]).boundary
    points = T.sample_grid(d=15)
    assert all(T._contains(points))


def test_triangle_boundary_grid_sampling_with_n_and_variable_origin():
    T = Triangle(R2('x'), origin, [5, 0], [0, 1]).boundary
    time = Points(torch.tensor([[0.0], [1.0], [2.0]]), R1('t'))
    points = T.sample_grid(n=50, params=time)
    assert points.as_tensor.shape == (150, 2)
    times = Points(torch.repeat_interleave(time, 50, dim=0), R1('t'))
    assert all(T._contains(points, times))


def test_triangle_boundary_grid_sampling_with_n_and_variable_corners():
    T = Triangle(R2('x'), origin, vec_1, vec_2).boundary
    time = Points(torch.tensor([[0.0], [1.0], [2.0]]), R1('t'))
    points = T.sample_grid(n=100, params=time)
    assert points.as_tensor.shape == (300, 2)
    times = Points(torch.repeat_interleave(time, 100, dim=0), R1('t'))
    assert all(T._contains(points, times))