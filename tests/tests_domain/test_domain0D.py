import pytest
import torch
import numpy as np

from torchphysics.problem.domains.domain import BoundaryDomain, Domain
from torchphysics.problem.domains.domain0D.point import Point
from torchphysics.problem.spaces.space import R1, R2, R3
from torchphysics.problem.spaces.points import Points
from torchphysics.utils.user_fun import DomainUserFunction


def p(t):
    return 2*t


def p2(t):
    if not isinstance(t, torch.Tensor):
        return [2*t, 0]
    return torch.column_stack((2*t, torch.zeros_like(t)))


def test_create_domain():
    d = Domain(R2('x'))
    assert d.dim == 2
    assert d._user_volume is None


def test_input_transform():
    d = Point(R1('x'), 4)
    bound = BoundaryDomain(d)
    points = torch.tensor([[0], [2.0]])
    new_points, params, device = \
        bound._transform_input_for_normals(points, Points.empty(), 0)
    assert torch.equal(new_points.as_tensor, points)
    assert len(params) == 0
    assert device.type == 'cpu'


def test_input_transform_with_params():
    d = Point(R1('x'), 4)
    bound = BoundaryDomain(d)
    points = torch.tensor([[0], [2.0]])
    params = torch.tensor([[0], [2.0]])
    new_points, new_params, device = \
        bound._transform_input_for_normals(points, {'x': params}, 0)
    assert torch.equal(new_points.as_tensor, points)
    assert torch.equal(new_params.as_tensor, params)
    assert device.type == 'cpu'


def test_create_point():
    P = Point(R1('x'), 4)
    assert P.dim == 0
    assert 'x' in P.space
    assert P.point.fun == 4


def test_create_point_in_higher_dim():
    P = Point(R2('x'), [4, 0])
    assert P.dim == 0
    assert 'x' in P.space
    assert P.point.fun[0] == 4
    assert P.point.fun[1] == 0


def test_create_point_with_variable_point():
    P = Point(R1('x'), DomainUserFunction(p))
    assert P.dim == 0
    assert 'x' in P.space
    assert P.point.fun == p


def test_call_point():
    P = Point(R2('x'), p2)
    called_P = P(t=3)
    assert called_P.dim == 0
    assert 'x' in called_P.space
    assert called_P.point.fun[0] == 6
    assert called_P.point.fun[1] == 0


def test_point_volume():
    P = Point(R1('x'), 0.0)
    assert P._get_volume() == 1


def test_point_has_no_boundary():
    P = Point(R2('x'), p2)
    with pytest.raises(NotImplementedError):
        P.boundary


def test_point_contains():
    P = Point(R1('x'), 4)
    points = Points(torch.tensor([[4.0], [0.0], [3.9]]), R1('x'))
    inside = P._contains(points)
    assert inside[0]
    assert not any(inside[1:])


def test_point_contains_over_in():
    P = Point(R1('x'), 4)
    assert Points(torch.tensor([[4.0]]), R1('x')) in P


def test_point_contains_if_point_variable():
    P = Point(R1('x'), p)
    points = Points(torch.tensor([[4.0], [0.0], [3.9]]), R1('x'))
    time = Points(torch.tensor([[2.0], [0.0], [-1.0]]), R1('t'))
    inside = P._contains(points, time)
    assert inside.shape == (3, 1)
    assert all(inside[:2])
    assert not any(inside[2])   


def test_point_contains_in_higher_dim():
    P = Point(R3('y'), [4.0, 0.0, 3.0])
    points = Points(torch.tensor([[4.0, 0.0, 3.0], [0.0, 0.0, 3.0],
                                  [3.9, 2.0, 8.0], [4.0, 1.0, 3.0]]), R3('y'))
    inside = P._contains(points)
    assert inside.shape == (4, 1)
    assert inside[:1]
    assert not any(inside[1:])


def test_point_contains_higher_dim_and_variable():
    P = Point(R2('x'), p2)
    points = Points(torch.tensor([[1.0, 0.0, 0.5], [3.0, 1.0, 1.0],
                                  [-1.0, 0.0, 0.0], [2.0, 5.0, 1.0]]), R2('x')*R1('t'))
    inside = P._contains(points)
    assert inside[0]
    assert not any(inside[1:])


def test_point_bounding_box():
    P = Point(R1('x'), 4)
    bounds = P.bounding_box()
    assert len(bounds) == 2
    assert bounds[0] == 3.9
    assert bounds[1] == 4.1


def test_point_bounding_box_higher_dim():
    P = Point(R1('x')*R2('y'), [4, 3, 1])
    bounds = P.bounding_box()
    assert len(bounds) == 6
    assert bounds[0] == 3.9
    assert bounds[1] == 4.1
    assert bounds[2] == 2.9
    assert bounds[3] == 3.1
    assert bounds[4] == 0.9
    assert bounds[5] == 1.1


def test_point_bounding_box_moving_point():
    P = Point(R1('x'), p)
    bounds = P.bounding_box(Points(torch.tensor([[1.0], [9.0], [2]]), R1('t')))
    assert len(bounds) == 2
    assert bounds[0] == 2
    assert bounds[1] == 18


def test_point_bounding_box_moving_point_higher_dim():
    P = Point(R2('x'), p2)
    bounds = P.bounding_box(Points(torch.tensor([[1.0], [9.0], [2]]), R1('t')))
    assert len(bounds) == 4
    assert bounds[0] == 2
    assert bounds[1] == 18
    assert np.isclose(bounds[2], -0.1)
    assert np.isclose(bounds[3], 0.1)


def test_point_random_sampling_with_n():
    P = Point(R1('x'), 4)
    points = P.sample_random_uniform(n=25).as_tensor
    assert points.shape == (25, 1)
    assert all(torch.isclose(points, torch.tensor(4.0)))


def test_point_random_sampling_with_d():
    P = Point(R1('x'), 4)
    points = P.sample_random_uniform(d=13).as_tensor
    assert all(torch.isclose(points, torch.tensor(4.0)))


def test_point_random_sampling_with_higher_dim():
    P = Point(R3('x'), [4.0, 0.0, 1.3])
    points = P.sample_random_uniform(n=25).as_tensor
    assert points.shape == (25, 3)
    assert all(torch.isclose(points[:, 0], torch.tensor(4.0)))
    assert all(torch.isclose(points[:, 1], torch.tensor(0.0)))
    assert all(torch.isclose(points[:, 2], torch.tensor(1.3)))


def test_point_random_sampling_with_n_moving_point():
    P = Point(R1('x'), p)
    time = Points(torch.tensor([[1.0], [0.0]]), R1('t'))
    points = P.sample_random_uniform(n=25, params=time).as_tensor
    assert points.shape == (50, 1)
    assert all(torch.isclose(points[:25], torch.tensor(2.0)))
    assert all(torch.isclose(points[25:], torch.tensor(0.0)))


def test_point__cant_random_sample_with_d_at_different_moments():
    P = Point(R1('x'), p)
    time = Points(torch.tensor([[1.0], [0.0]]), R1('t'))
    with pytest.raises(ValueError):
        P.sample_random_uniform(d=25, params=time).as_tensor


def test_point_random_sampling_with_n_moving_point_higher_dim():
    P = Point(R2('x'), p2)
    time = Points(torch.tensor([[1.0], [0.0]]), R1('t'))
    points = P.sample_random_uniform(n=5, params=time).as_tensor
    assert points.shape == (10, 2)
    assert all(torch.isclose(points[:5, 0], torch.tensor(2.0)))
    assert all(torch.isclose(points[5:, 0], torch.tensor(0.0)))
    assert all(torch.isclose(points[:, 1], torch.tensor(0.0)))


def test_point_grid_sampling_with_n():
    P = Point(R1('x'), 4)
    points = P.sample_grid(n=25).as_tensor
    assert points.shape == (25, 1)
    assert all(torch.isclose(points, torch.tensor(4.0)))