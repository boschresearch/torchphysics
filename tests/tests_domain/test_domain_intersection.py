import pytest
import torch
import numpy as np

from torchphysics.problem.domains.domain2D.circle import Circle
from torchphysics.problem.domains.domain2D.parallelogram import Parallelogram
from torchphysics.problem.domains.domain1D.interval import Interval
from torchphysics.problem.domains.domain3D.sphere import Sphere
from torchphysics.problem.spaces.space import R2, R1, R3
from torchphysics.problem.spaces.points import Points


def test_make_intersection():
    C1 = Circle(R2('x'), [0, 0], 3)
    C2 = Circle(R2('x'), [3, 0], 1)
    C1 & C2


def test_cant_intersection_from_different_spaces():
    C1 = Circle(R2('x'), [0, 0], 3)
    C2 = Circle(R2('y'), [3, 0], 1)
    with pytest.raises(ValueError):
        C1 & C2


def test_intersection_register_nessecary_variables():
    C1 = Circle(R2('x'), [0, 0], lambda t: t+1)
    C2 = Circle(R2('x'), [0, 0], lambda u: u+3)
    C = C1 & C2
    assert 'u' in C.necessary_variables 
    assert 't' in C.necessary_variables 


def test_call_intersection_domain():
    C1 = Circle(R2('x'), [0, 0], lambda t: t+1)
    C2 = Circle(R2('x'), [0, 0], 5)
    C = C2 & C1
    C(t=2)


def test_intersection_contains_domain_constant_in_2D():
    C1 = Circle(R2('x'), [0, 0], 5)
    C2 = Circle(R2('x'), [0, 0], 2)
    C = C1 & C2  
    points = Points(torch.tensor([[0.0, 0.0], [0, -1], [0.5, -0.1],
                                  [4.1, 0], [-0.1, -3]]), R2('x'))
    inside = C._contains(points)
    assert not any(inside[3:])
    assert all(inside[:3])


def test_intersection_contains_domain_changes_in_2D():
    C1 = Circle(R2('x'), [0, 0], 5)
    C2 = Circle(R2('x'), [0, 0], lambda t: t+1)
    C = C1 & C2  
    points = Points(torch.tensor([[0.0, 0.0], [0, -1], [0.5, -0.1],
                                  [3.1, 0], [-0.1, -3], [0.0, 0.0], [0, -1],
                                  [0.5, -0.1], [2.8, 0], [-0.1, 4.5]]), R2('x'))
    time = Points(torch.tensor([[1.0], [3.0]]).repeat_interleave(5, dim=0), R1('t'))
    inside = C._contains(points, time)
    assert not any(inside[3:5])
    assert all(inside[:3])
    assert not inside[9]
    assert all(inside[5:9])


def test_intersection_contains_domain_in_1D():
    C1 = Interval(R1('x'), 0, 1)
    C2 = Interval(R1('x'), 0.5, 0.6)
    C = C1 & C2  
    points = Points(torch.tensor([[0.0], [0.45], [0.6991],
                                  [0.501], [0.576]]), R1('x'))
    inside = C._contains(points)
    assert not any(inside[:3])
    assert all(inside[3:])


def test_intersection_contains_domain_in_3D():
    C1 = Sphere(R3('x'), [0, 0, 0], 3)
    C2 = Sphere(R3('x'), [0, 0, 0], 1)
    C = C1 & C2  
    points = Points(torch.tensor([[0.0, 0.3, 0.0], [0.1, 0.2, -0.1], [2.001, 0, 0],
                                  [-10, 0, 0], [1, 1, 0.03]]), R3('x'))
    inside = C._contains(points)
    assert not any(inside[2:])
    assert all(inside[:2])


def test_get_volume_of_intersection():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Parallelogram(R2('x'), [-0.5, -0.5], [0.5, 0], [0, 0.5])
    P = P1 & P2
    assert torch.isclose(P._get_volume(), P1._get_volume())


def test_get_volume_of_intersection_if_domain_changes():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    C2 = Circle(R2('x'), [0.5, 0.5], lambda t: 0.1*t)
    P = P1 & C2
    time = Points(torch.tensor([[1.0], [2.0]]), R1('t'))
    assert torch.allclose(P._get_volume(time), P1._get_volume(time))


def test_bounding_box_of_intersection():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    C2 = Circle(R2('x'), [0, 0], 0.3)
    P = P1 & C2
    assert np.allclose(P.bounding_box(), [0, 0.3, 0, 0.3])


def test_sample_random_uniform_in_intersection_with_n():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Parallelogram(R2('x'), [-0.5, -0.5], [0.5, 0], [0, 0.5])
    P = P1 & P2
    points = P.sample_random_uniform(n=20)
    assert len(points) == 20
    assert torch.all(P._contains(points))


def test_sample_random_uniform_in_intersection_with_n_with_params():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Circle(R2('x'), [1, 1], lambda t:t+1)
    P = P1 & P2
    time = Points(torch.tensor([[1.0], [1.5]]), R1('t'))
    points = P.sample_random_uniform(n=20, params=time, device='cpu')
    assert len(points) == 40


def test_sample_random_uniform_in_intersection_with_d():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Parallelogram(R2('x'), [-1, 0], [0.5, 0], [-1, 1])
    P = P1 & P2
    points = P.sample_random_uniform(d=12)
    assert torch.all(points.as_tensor[:, 0] <= 0.5)
    assert torch.all(points.as_tensor[:, 0] >= 0.0)


def test_sample_random_uniform_in_intersection_with_d_in_1D():
    I1 = Interval(R1('x'), 0, 1)
    I2 = Interval(R1('x'), 0.5, 1)
    I = I1 & I2
    points = I.sample_random_uniform(d=102)
    assert torch.all(points.as_tensor >= 0.5)
    assert torch.all(I1._contains(points))


def test_sample_grid_in_intersection_with_n():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Parallelogram(R2('x'), [-0.5, -0.5], [0.5, 0], [0, 0.5])
    P = P1 & P2
    points = P.sample_grid(n=120)
    assert len(points) == 120
    assert torch.all(P._contains(points))



def test_sample_grid_in_intersection_with_d():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Parallelogram(R2('x'), [0, 0], [0.5, 0], [0, 1])
    P = P1 & P2
    points = P.sample_grid(d=12)
    assert torch.all(points.as_tensor[:, 0] <= 0.5)
    assert torch.all(points.as_tensor[:, 0] >= 0.0)


def test_sample_grid_in_intersection_with_d_in_1D():
    I1 = Interval(R1('x'), 0, 1)
    I2 = Interval(R1('x'), 0.3, 2)
    I = I1 & I2
    points = I.sample_grid(d=102)
    assert torch.all(points.as_tensor >= 0.3)
    assert torch.all(points.as_tensor <= 1)
    assert torch.all(I1._contains(points))


def test_get_intersection_boundary():
    C1 = Circle(R2('x'), [0, 0], 3)
    C2 = Circle(R2('x'), [0, 0], 1)
    C = C1 & C2
    C.boundary


def test_intersection_boundary_contains_domain_constant_in_2D():
    C1 = Parallelogram(R2('x'), [0, 0], [2, 0], [0, 2])
    C2 = Circle(R2('x'), [0, 0], 2)
    C = C1 & C2 
    C = C.boundary 
    points = Points(torch.FloatTensor([[0.0, 0.0], [0, 1], [0, 2],
                                  [2.0, 0], [2*np.cos(0.2), 2*np.sin(0.2)],
                                  [0, -2], [2, 2.0]]), R2('x'))
    inside = C._contains(points)
    assert not any(inside[5:])
    assert all(inside[:5])


def test_intersection_boundary_contains_domain_changes_in_2D():
    C1 = Parallelogram(R2('x'), [0, 0], [2, 0], [0, 2])
    C2 = Circle(R2('x'), [0, 0], lambda t: t+1)
    C = C1 & C2  
    C = C.boundary 
    points = Points(torch.FloatTensor([[0.0, 0.0], [0, 1], [0, 2],
                                       [2.0, 0], [2*np.cos(0.2), 2*np.sin(0.2)],
                                       [0, -2], [2, 2.0], [0, 0], [1, 0], 
                                       [0.5, 0], [np.cos(0.4), np.sin(0.4)], [0, 0.2],
                                       [-1, 0], [1.8, 0]]), R2('x'))
    time = Points(torch.tensor([[1.0], [0.0]]).repeat_interleave(7, dim=0), R1('t'))
    inside = C._contains(points, time)
    assert not any(inside[5:7])
    assert all(inside[:5])
    assert not any(inside[12:])
    assert all(inside[7:12])


def test_intersection_boundary_contains_domain_in_1D():
    C1 = Interval(R1('x'), 0, 1)
    C2 = Interval(R1('x'), 0.5, 1)
    C = C1 & C2  
    C = C.boundary 
    points = Points(torch.tensor([[0.0], [-1], [0.5001],
                                  [0.5], [1.0]]), R1('x'))
    inside = C._contains(points)
    assert not any(inside[:3])
    assert all(inside[3:])


def test_intersection_boundary_contains_domain_in_3D():
    C1 = Sphere(R3('x'), [0, 0, 0], 3)
    C2 = Sphere(R3('x'), [0, 0, 0], 1)
    C = C1 & C2  
    C = C.boundary 
    points = Points(torch.tensor([[0.0, 1.0, 0.0], [-1, 0, 0], [0, 1, 0],
                                  [3, 0, 0], [0.8, 0, 0.03]]), R3('x'))
    inside = C._contains(points)
    assert not any(inside[3:])
    assert all(inside[:3])


def test_get_volume_of_intersection_boundary():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    C2 =  Parallelogram(R2('x'), [0, 0], [0.5, 0], [0, 1])
    P = P1 & C2
    P = P.boundary 
    with pytest.warns(UserWarning):
        volume = P._get_volume()
    assert torch.isclose(volume, torch.tensor(7.0))


def test_get_volume_of_intersection_boundary_if_domain_changes():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    C2 = Circle(R2('x'), [0.5, 0.5], lambda t: 0.1*t)
    P = P1 & C2
    P = P.boundary
    time = Points(torch.tensor([[1.0], [2.0]]), R1('t'))
    with pytest.warns(UserWarning):
        volume = P._get_volume(time)
    assert torch.allclose(volume,
                          torch.tensor([[4 + 0.2*np.pi], [4 + 0.4*np.pi]]))


def test_bounding_box_of_intersection_boundary():
    I1 = Interval(R1('x'), 0, 1)
    I2 = Interval(R1('x'), 0.5, 1)
    I = I1 & I2
    I = I.boundary
    assert all(np.equal(I.bounding_box(), [0.5, 1.0]))


def test_sample_random_uniform_in_intersection_boundary_with_n():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Parallelogram(R2('x'), [-0.5, -0.5], [0.5, 0], [0, 0.5])
    P = (P1 & P2)
    points = P.boundary.sample_random_uniform(n=25)
    assert len(points) == 25
    assert torch.all(P.boundary._contains(points))


def test_sample_random_uniform_in_intersection_boundary_with_d():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Parallelogram(R2('x'), [0, 0], [0.5, 0], [0, 1])
    P = P1 & P2
    points = P.boundary.sample_random_uniform(d=12)
    assert torch.all(points.as_tensor[:, 0] <= 0.5)


def test_sample_random_uniform_in_intersection_boundary_with_d_in_1D():
    I1 = Interval(R1('x'), 0, 1)
    I2 = Interval(R1('x'), 0.5, 1)
    I = I1 & I2
    points = I.boundary.sample_random_uniform(d=102)
    assert torch.all(points.as_tensor >= 0.5)


def test_sample_grid_in_intersection_boundary_with_n():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Parallelogram(R2('x'), [-0.5, -0.5], [0.5, 0], [0, 0.5])
    P = P1 & P2
    points = P.boundary.sample_grid(n=50)
    assert len(points) == 50
    assert torch.all(P.boundary._contains(points))


def test_sample_grid_in_intersection_boundary_with_d():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Parallelogram(R2('x'), [0, 0], [0.5, 0], [0, 1])
    P = P1 & P2
    points = P.boundary.sample_grid(d=12)
    assert torch.all(points.as_tensor[:, 0] <= 0.5)
    assert torch.all(P.boundary._contains(points))


def test_sample_grid_in_intersection_boundary_with_d_in_1D():
    I1 = Interval(R1('x'), 0, 1)
    I2 = Interval(R1('x'), 0.3, 1)
    I = I1 & I2
    points = I.boundary.sample_grid(d=102)
    assert torch.all(points.as_tensor >= 0.3)
    assert torch.all(I.boundary._contains(points))


def test_get_normals_of_intersection_in_2D():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Parallelogram(R2('x'), [-1, 0], [0.5, 0], [-1, 1])
    P = P1 & P2
    points = Points(torch.tensor([[0.3, 0.0], [0.5, 0.5], [0.2, 1], 
                                  [0.0, 0.8], [0.0, 0.5]]), R2('x'))
    normals = P.boundary.normal(points)
    expected_normals = torch.tensor([[0, -1.0], [1, 0], [0, 1], [-1, 0], 
                                     [-1, 0]])
    assert torch.allclose(normals, expected_normals)


def test_get_normals_of_intersection_in_1D():
    I1 = Interval(R1('x'), 0, 1)
    I2 = Interval(R1('x'), 0.3, 1)
    I = I1 & I2
    points = Points(torch.tensor([[1.0], [0.3]]), R1('x'))
    normals = I.boundary.normal(points)
    expected_normals = torch.tensor([[1], [-1]])
    assert torch.allclose(normals, expected_normals)