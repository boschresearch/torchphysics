import pytest
import torch

from torchphysics.problem.domains.domain1D.interval import (Interval,
                                                           IntervalBoundary, 
                                                           IntervalSingleBoundaryPoint)
from torchphysics.problem.spaces.space import R1
from torchphysics.problem.spaces.points import Points


def lower_bound(t):
    return -t

def upper_bound(t):
    return 2*t + 1


def test_create_interval():
    I = Interval(R1('x'), 0, 1)
    assert I.lower_bound.fun == 0
    assert I.upper_bound.fun == 1
    assert 'x' in I.space


def test_create_interval_with_variable_bounds():
    I = Interval(R1('x'), lower_bound=lower_bound, upper_bound=upper_bound)
    assert I.lower_bound.fun == lower_bound
    assert I.upper_bound.fun == upper_bound


def test_create_interval_mixed_bounds():
    I = Interval(R1('x'), lower_bound=0, upper_bound=upper_bound)
    assert I.lower_bound.fun == 0
    assert I.upper_bound.fun == upper_bound
    I = Interval(R1('x'), lower_bound=lower_bound, upper_bound=1)
    assert I.lower_bound.fun == lower_bound
    assert I.upper_bound.fun == 1


def test_call_interval():
    I = Interval(R1('x'), lower_bound=0, upper_bound=upper_bound)
    called_I = I(t=2)
    assert called_I.lower_bound.fun == 0
    assert called_I.upper_bound.fun == 5


def test_bounding_box_interval():
    I = Interval(R1('x'), 0, 1)
    bounds = I.bounding_box()
    assert bounds[0] == 0
    assert bounds[1] == 1


def test_bounding_box_interval_variable_bounds():
    I = Interval(R1('x'), lower_bound=lower_bound, upper_bound=upper_bound)
    time = Points(torch.tensor([1, 2, 3, 4]).reshape(-1, 1), R1('t'))
    bounds = I.bounding_box(time)
    assert bounds[0] == -4
    assert bounds[1] == 9


def test_interval_contains():
    I = Interval(R1('x'), 0, 1)
    points = Points(torch.tensor([0.5, 0.7, 0, -2, -0.1]).reshape(-1, 1), R1('x'))
    inside = I._contains(points)
    assert all(inside[:3])
    assert not any(inside[3:])


def test_interval_volume():
    I = Interval(R1('x'), 0, 1)
    assert I._get_volume() == 1


def test_interval_volume_bounds_change():
    I = Interval(R1('x'), 0, lambda t : t+1)
    t = Points(torch.tensor([[2.], [1]]), R1('t'))
    assert all(I._get_volume(t) == torch.tensor([[3.0], [2.0]]))


def test_interval_contains_if_one_bound_changes():
    I = Interval(R1('x'), 0, upper_bound)
    points = Points(torch.tensor([0.5, 0, 7, -2, -0.1]).reshape(-1, 1), R1('x'))
    time = Points(torch.tensor([0, 0, 1, -2, -0.1]).reshape(-1, 1), R1('t'))
    inside = I._contains(points, time)
    assert all(inside[:2])
    assert not any(inside[2:])


def test_interval_contains_if_both_bound_changes():
    I = Interval(R1('x'), lower_bound, upper_bound)
    points = Points(torch.tensor([0.5, -1, 7, -2, -0.1]).reshape(-1, 1), R1('x'))
    time = Points(torch.tensor([0, 2, 1, -2, -0.1]).reshape(-1, 1), R1('t'))
    inside = I._contains(points, time)
    assert all(inside[:2])
    assert not any(inside[2:])


def test_interval_random_sampling_with_n():
    I = Interval(R1('x'), 0, 1)
    points = I.sample_random_uniform(n=10)
    assert points.as_tensor.shape == (10, 1)
    assert all(I._contains(points))


def test_interval_random_sampling_with_d():
    I = Interval(R1('x'), 0, 1)
    points = I.sample_random_uniform(d=11)
    assert all(I._contains(points))


def test_interval_grid_sampling_with_d():
    I = Interval(R1('x'), 0, 1)
    points = I.sample_grid(d=11)
    assert all(I._contains(points))


def test_interval_random_sampling_with_n_and_variable_bounds():
    I = Interval(R1('x'), 0, upper_bound)
    t = Points(torch.tensor([0, 1]).reshape(-1, 1), R1('t'))
    points = I.sample_random_uniform(n=4, params=t)
    assert points.as_tensor.shape == (8, 1)
    assert all(I._contains(points, Points(torch.repeat_interleave(t, 4, dim=0), R1('t'))))


def test_interval_grid_sampling_with_n():
    I = Interval(R1('x'), 0, 1)
    points = I.sample_grid(n=10)
    assert points.as_tensor.shape == (10, 1)
    assert all(I._contains(points))
    for i in range(8):
        dist_1 = torch.linalg.norm(points.as_tensor[i+1] - points.as_tensor[i])
        dist_2 = torch.linalg.norm(points.as_tensor[i+1] - points.as_tensor[i+2])
        assert torch.isclose(dist_1, dist_2)


def test_interval_grid_sampling_with_n_and_variable_bounds():
    I = Interval(R1('x'), 0, upper_bound)
    t = Points(torch.tensor([0, 1]).reshape(-1, 1), R1('t'))
    points = I.sample_grid(n=4, params=t)
    assert points.as_tensor.shape == (8, 1)
    assert all(I._contains(points, Points(torch.repeat_interleave(t, 4, dim=0), R1('t'))))


def test_get_Intervalboundary():
    I = Interval(R1('x'), 0, 1)
    boundary = I.boundary
    assert isinstance(boundary, IntervalBoundary)
    assert I == boundary.domain


def test_call_interval_boundary():
    I = Interval(R1('x'), 0, upper_bound).boundary
    new_I = I(t=2)
    assert isinstance(new_I, IntervalBoundary)
    assert new_I.domain.lower_bound.fun == 0
    assert new_I.domain.upper_bound.fun == 5


def test_interval_boundary_volume():
    I = Interval(R1('x'), 0, 1).boundary
    assert I._get_volume() == 2


def test_interval_boundary_contains():
    I = Interval(R1('x'), 0, 1).boundary
    points = Points(torch.tensor([0, 0, 1, -2, -0.1, 0.5]).reshape(-1, 1), R1('x'))
    inside = I._contains(points)
    assert all(inside[:3])
    assert not any(inside[3:])


def test_interval_boundary_contains_if_bound_changes():
    I = Interval(R1('x'), 0, upper_bound).boundary
    points = Points(torch.tensor([0, 1, 0, 4, -1, 12.0]).reshape(-1, 1), R1('x'))
    time = Points(torch.tensor([0, 0, 1, 1, 1, 2.0]).reshape(-1, 1), R1('t'))
    inside = I._contains(points, params=time)
    assert all(inside[:3])
    assert not any(inside[3:])


def test_interval_boundary_random_sampling_with_n():
    I = Interval(R1('x'), 0, 1).boundary
    points = I.sample_random_uniform(n=10)
    assert points.as_tensor.shape == (10, 1)
    assert all(I._contains(points))


def test_interval_boundary_random_sampling_with_d():
    I = Interval(R1('x'), 0, 1).boundary
    points = I.sample_random_uniform(d=13)
    assert all(I._contains(points))


def test_interval_boundary_grid_sampling_with_d():
    I = Interval(R1('x'), 0, 1).boundary
    points = I.sample_grid(d=13)
    assert all(I._contains(points))


def test_interval_boundary_grid_sampling_with_n():
    I = Interval(R1('x'), 0, 1).boundary
    points = I.sample_grid(n=10)
    assert points.as_tensor.shape == (10, 1)
    assert all(I._contains(points))
    for i in range(10):
        point_eq_0 = points.as_tensor[i] == 0
        point_eq_1 = points.as_tensor[i] == 1
        assert point_eq_0 or point_eq_1 


def test_interval_boundary_random_sampling_with_n_and_variable_bounds():
    I = Interval(R1('x'), 0, upper_bound).boundary
    time = Points(torch.tensor([0.0, 1.0, 2]).reshape(-1, 1), R1('t'))
    points = I.sample_random_uniform(n=2, params=time)
    assert points.as_tensor.shape == (6, 1)
    assert all(I._contains(points, Points(torch.repeat_interleave(time, 2, dim=0), R1('t'))))


def test_interval_boundary_grid_with_n_and_variable_bounds():
    I = Interval(R1('x'), 0, upper_bound).boundary
    time = Points(torch.tensor([0.0, 1.0]).reshape(-1, 1), R1('t'))
    points = I.sample_grid(n=4, params=time)
    assert points.as_tensor.shape == (8, 1)
    assert all(I._contains(points, Points(torch.repeat_interleave(time, 4, dim=0), R1('t'))))


def test_interval_normals():
    I = Interval(R1('x'), 0, 1).boundary
    points = Points(torch.tensor([0, 1.0, 0]).reshape(-1, 1), R1('x'))
    normals = I.normal(points)
    assert normals.shape == (3, 1)
    assert all(torch.isclose(torch.tensor([[-1], [1], [-1]]), normals))


def test_interval_normals_if_bounds_change():
    I = Interval(R1('x'), lower_bound, 1).boundary
    time = Points(torch.tensor([0, 0, 1, 1, 2.0]).reshape(-1, 1), R1('t'))
    points = Points(torch.tensor([0, 1.0, -1, 1, -2]).reshape(-1, 1), R1('x'))
    normals = I.normal(points, params=time)
    assert normals.shape == (5, 1)
    assert all(torch.isclose(torch.tensor([[-1], [1], [-1], [1], [-1]]), normals))


def test_interval_get_left_boundary():
    I = Interval(R1('x'), 0, 1).boundary_left
    assert isinstance(I, IntervalSingleBoundaryPoint)
    assert I.side.fun == 0


def test_interval_get_right_boundary():
    I = Interval(R1('x'), 0, 1).boundary_right
    assert isinstance(I, IntervalSingleBoundaryPoint)
    assert I.side.fun == 1


def test_call_single_interval_bound():
    I = Interval(R1('x'), 0, upper_bound).boundary_right
    called_I = I(t=3)
    assert isinstance(called_I, IntervalSingleBoundaryPoint)
    assert called_I.side.fun == upper_bound


def test_single_interval_bound_contains():
    I = Interval(R1('x'), 0, 4).boundary_right
    points = Points(torch.tensor([[4.0], [0.0], [3.9]]), R1('x'))
    inside = I._contains(points)
    assert inside[0]
    assert not any(inside[1:])


def test_single_interval_bound_contains_if_bound_variable():
    I = Interval(R1('x'), 0, upper_bound).boundary_right
    points = Points(torch.tensor([[5.0], [1.0], [3.9]]), R1('x'))
    time = Points(torch.tensor([[2.0], [0.0], [-1.0]]), R1('t'))
    inside = I._contains(points, params=time)
    assert inside.shape == (3, 1)
    assert all(inside[:2])
    assert not any(inside[2])   


def test_single_interval_bound_bounding_box():
    I = Interval(R1('x'), 0, 4).boundary_right
    bounds = I.bounding_box()
    assert len(bounds) == 2
    assert bounds[0] == 0
    assert bounds[1] == 4


def test_single_interval_bound_random_sampling_with_n():
    I = Interval(R1('x'), 0, 4).boundary_left
    points = I.sample_random_uniform(n=25)
    assert points.as_tensor.shape == (25, 1)
    assert all(torch.isclose(points.as_tensor, torch.tensor(0.0)))


def test_single_interval_bound_random_sampling_with_d():
    I = Interval(R1('x'), 0, 4).boundary_left
    points = I.sample_random_uniform(d=11)
    assert all(torch.isclose(points.as_tensor, torch.tensor(0.0)))


def test_single_interval_bound_random_sampling_with_n_moving_bound():
    I = Interval(R1('x'), lower_bound, 4).boundary_left
    time = Points(torch.tensor([[1.0], [0.0]]), R1('t'))
    points = I.sample_random_uniform(n=25, params=time)
    assert points.as_tensor.shape == (50, 1)
    assert all(torch.isclose(points.as_tensor[:25], torch.tensor(-1.0)))
    assert all(torch.isclose(points.as_tensor[25:], torch.tensor(0.0)))


def test_single_interval_bound_grid_sampling_with_n():
    I = Interval(R1('x'), 0, 4).boundary_left
    points = I.sample_grid(n=25)
    assert points.as_tensor.shape == (25, 1)
    assert all(torch.isclose(points.as_tensor, torch.tensor(0.0)))


def test_interval_normals_left_side():
    I = Interval(R1('x'), 0, 1).boundary_left
    points = Points(torch.tensor([0, 0.0, 0]).reshape(-1, 1), R1('x'))
    normals = I.normal(points)
    assert normals.shape == (3, 1)
    assert all(torch.isclose(torch.tensor([[-1], [-1.0], [-1]]), normals))


def test_interval_normals_ride_side():
    I = Interval(R1('x'), 0, 1).boundary_right
    points = Points(torch.tensor([1, 1.0, 1]).reshape(-1, 1), R1('x'))
    normals = I.normal(points)
    assert normals.shape == (3, 1)
    assert all(torch.isclose(torch.tensor([[1], [1.0], [1]]), normals))


def test_interval_boundary_left_volume():
    I = Interval(R1('x'), 0, 1).boundary_left
    assert I._get_volume() == 1