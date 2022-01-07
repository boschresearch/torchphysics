import pytest
import torch
import numpy as np

from torchphysics.problem.domains.domain2D.circle import Circle
from torchphysics.problem.domains.domain2D.parallelogram import Parallelogram
from torchphysics.problem.domains.domain1D.interval import Interval
from torchphysics.problem.domains.domain3D.sphere import Sphere
from torchphysics.problem.spaces.space import R2, R1, R3
from torchphysics.problem.spaces.points import Points


def test_make_union():
    C1 = Circle(R2('x'), [0, 0], 3)
    C2 = Circle(R2('x'), [0, 0], 1)
    C1 + C2


def test_cant_union_from_different_spaces():
    C1 = Circle(R2('x'), [0, 0], 3)
    C2 = Circle(R2('y'), [0, 0], 1)
    with pytest.raises(ValueError):
        C1 + C2


def test_union_register_nessecary_variables():
    C1 = Circle(R2('x'), [0, 0], lambda t: t+1)
    C2 = Circle(R2('x'), [0, 0], lambda u: u+3)
    C = C1 + C2
    assert 'u' in C.necessary_variables 
    assert 't' in C.necessary_variables 


def test_call_union_domain():
    C1 = Circle(R2('x'), [0, 0], lambda t: t+1)
    C2 = Circle(R2('x'), [0, 0], 5)
    C = C2 + C1
    C(t=2)


def test_union_contains_domain_constant_in_2D():
    C1 = Circle(R2('x'), [0, 0], 5)
    C2 = Circle(R2('x'), [0, 0], 2)
    C = C1 + C2  
    points = Points(torch.tensor([[0.0, 0.0], [0, -1], [0.5, -0.1],
                                  [4.1, 0], [-0.1, -3]]), R2('x'))
    inside = C._contains(points)
    assert all(inside)


def test_union_contains_domain_changes_in_2D():
    C1 = Circle(R2('x'), [10, 0], 5)
    C2 = Circle(R2('x'), [0, 0], lambda t: t+1)
    C = C1 + C2  
    points = Points(torch.tensor([[0.0, 0.0], [0, -1], [0.5, -0.1],
                                  [3, 0], [10, -1], [0.0, 0.0], [0, -1],
                                  [0.5, -0.1], [3.1, 0], [10, 1]]), R2('x'))
    time = Points(torch.tensor([[1.0], [3.0]]).repeat_interleave(5, dim=0), R1('t'))
    inside = C._contains(points, time)
    assert all(inside[:3])
    assert not inside[3]
    assert all(inside[4:])



def test_union_contains_domain_in_1D():
    C1 = Interval(R1('x'), 0, 1)
    C2 = Interval(R1('x'), 1, 2)
    C = C1 + C2  
    points = Points(torch.tensor([[0.0], [0.45], [0.5001],
                                  [1.2], [1.8], [2.3], [-0.01]]), R1('x'))
    inside = C._contains(points)
    assert not any(inside[5:])
    assert all(inside[:5])


def test_union_contains_domain_in_3D():
    C1 = Sphere(R3('x'), [0.5, 0, 0], 3)
    C2 = Sphere(R3('x'), [0, 0, 0], 1)
    C = C1 + C2  
    points = Points(torch.tensor([[0.0, 2.0, 0.0], [1.5, 0.2, 0], [0.5001, 0, 0],
                                  [0.5, 2, 0], [-40, 0, 0.03]]), R3('x'))
    inside = C._contains(points)
    assert not any(inside[4:])
    assert all(inside[:4])


def test_get_volume_of_union_if_disjoint():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Parallelogram(R2('x'), [10, 0], [15, 0], [10, 5])
    P = P1 + P2
    P.disjoint = True
    assert torch.isclose(P._get_volume(), torch.tensor(26.0))


def test_get_volume_of_union_return_single_volumes():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Parallelogram(R2('x'), [10, 0], [15, 0], [10, 5])
    P = P1 + P2
    P.disjoint = True
    volume, volume_1, volume_2 = P._get_volume(return_value_of_a_b=True)
    assert torch.isclose(volume, torch.tensor(26.0))
    assert torch.isclose(volume_2, torch.tensor(25.0))
    assert torch.isclose(volume_1, torch.tensor(1.0))


def test_get_volume_of_union_if_not_disjoint():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Parallelogram(R2('x'), [-0.5, 0], [0.5, 0], [-0.5, 0.5])
    P = P1 + P2
    with pytest.warns(UserWarning):
        volume = P._get_volume()
    assert torch.isclose(volume, torch.tensor(1.5))


def test_get_volume_of_union_if_domain_changes():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    C2 = Circle(R2('x'), [5, 5], lambda t: 0.1*t)
    P = P1 + C2
    P.disjoint = True
    time = Points(torch.tensor([[1.0], [2.0]]), R1('t'))
    assert torch.allclose(P._get_volume(time),
                          torch.tensor([[1 + np.pi/100], [1 + np.pi/25]]))


def test_bounding_box_of_union():
    I1 = Interval(R1('x'), 0, 1)
    I2 = Interval(R1('x'), 0.5, 2)
    I = I1 + I2
    assert all(np.equal(I.bounding_box(), [0.0, 2.0]))


def test_sample_random_uniform_in_union_with_n():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Parallelogram(R2('x'), [-0.5, -0.5], [0.5, 0], [0, 0.5])
    P = P1 + P2
    points = P.sample_random_uniform(n=50)
    assert points.as_tensor.shape == (50, 2)


def test_sample_random_uniform_in_union_with_d():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Parallelogram(R2('x'), [-1, -1], [0.5, 0], [0, 1])
    P = P1 + P2
    points = P.sample_random_uniform(d=12)
    assert torch.all(P._contains(points))


def test_sample_random_uniform_in_union_with_d_in_1D():
    I1 = Interval(R1('x'), 0, 1)
    I2 = Interval(R1('x'), 1.5, 2)
    I = I1 + I2
    points = I.sample_random_uniform(d=102)
    assert torch.all(points.as_tensor <= 2.0)
    assert torch.all(points.as_tensor >= 0.0)


def test_sample_grid_in_union_with_n():
    P1 = Parallelogram(R2('x'), [5, 5], [6, 5], [5, 6])
    P2 = Parallelogram(R2('x'), [-0.5, -0.5], [0.5, 0], [0, 0.5])
    P = P1 + P2
    P.disjoint = True
    points = P.sample_grid(n=56)
    assert points.as_tensor.shape == (56, 2)


def test_sample_grid_in_union_with_n_and_one_small_domain():
    P1 = Parallelogram(R2('x'), [5, 5], [6, 5], [5, 6])
    P2 = Parallelogram(R2('x'), [0, 0], [0.01, 0], [0, 0.01])
    P = P1 + P2
    P.disjoint = True
    points = P.sample_grid(n=56)
    assert points.as_tensor.shape == (56, 2)


def test_sample_grid_in_union_with_d():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Parallelogram(R2('x'), [0, 0], [0.5, 0], [0, 1])
    P = P1 + P2
    points = P.sample_grid(d=12)
    assert torch.all(P._contains(points))


def test_sample_grid_in_union_with_d_in_1D():
    I1 = Interval(R1('x'), 0, 1)
    I2 = Interval(R1('x'), -0.3, 0.1)
    I = I1 + I2
    points = I.sample_grid(d=102)
    assert torch.all(I._contains(points))


def test_get_union_boundary():
    C1 = Circle(R2('x'), [0, 0], 3)
    C2 = Circle(R2('x'), [0, 0], 1)
    C = C1 + C2
    C.boundary


def test_union_boundary_contains_domain_constant_in_2D():
    C1 = Parallelogram(R2('x'), [0, 0], [2, 0], [0, 2])
    C2 = Circle(R2('x'), [0, 0], 2)
    C = C1 + C2 
    C = C.boundary 
    points = Points(torch.tensor([[0.0, -2.0], [-2.0, 0], [2, 0],
                                  [2, 2], [0, 0], [0, 1], [-5, 0]]), R2('x'))
    inside = C._contains(points)
    assert not any(inside[4:])
    assert all(inside[:4])


def test_union_boundary_contains_domain_changes_in_2D():
    C1 = Parallelogram(R2('x'), [0, 0], [2, 0], [0, 2])
    C2 = Circle(R2('x'), [0, 0], lambda t: t+1)
    C = C1 + C2  
    C = C.boundary 
    points = Points(torch.tensor([[0.0, -2.0], [-2.0, 0], [2, 0],
                                  [2, 2], [0, 0], [0, 1], [-5, 0], 
                                  [-1, 0], [0, -1], [0, 2], [2, 0], 
                                  [-2, 0], [0.0, 0.0], [0, 0.3]]), R2('x'))
    time = Points(torch.tensor([[1.0], [0.0]]).repeat_interleave(7, dim=0), R1('t'))
    inside = C._contains(points, time)
    assert not any(inside[4:7])
    assert all(inside[:4])
    assert not any(inside[11:])
    assert all(inside[7:11])

def test_union_boundary_contains_domain_in_1D():
    C1 = Interval(R1('x'), 0, 1)
    C2 = Interval(R1('x'), -1, -0.3)
    C = C1 + C2  
    C = C.boundary 
    points = Points(torch.tensor([[1.0], [-1], [-0.3],
                                  [0.0], [-0.2]]), R1('x'))
    inside = C._contains(points)
    assert not any(inside[4:])
    assert all(inside[:4])


def test_get_volume_of_union_boundary_if_disjoint():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    C2 = Circle(R2('x'), [5, 5], 0.2)
    P = P1 + C2
    P.disjoint = True
    P = P.boundary 
    assert torch.isclose(P._get_volume(), torch.tensor(4+0.4*np.pi))


def test_get_volume_of_union_boundary_if_not_disjoint():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Circle(R2('x'), [5, 5], 0.2)
    P = P1 + P2
    P = P.boundary
    with pytest.warns(UserWarning):
        volume = P._get_volume()
    assert torch.isclose(volume, torch.tensor(4+0.4*np.pi))


def test_get_volume_of_union_boundary_if_domain_changes():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    C2 = Circle(R2('x'), [5, 5], lambda t: 0.1*t)
    P = P1 + C2
    P.disjoint = True
    P = P.boundary
    time = Points(torch.tensor([[1.0], [2.0]]), R1('t'))
    assert torch.allclose(P._get_volume(time),
                          torch.tensor([[4 + 0.2*np.pi], [4 + 0.4*np.pi]]))


def test_bounding_box_of_union_boundary():
    I1 = Interval(R1('x'), 0, 1)
    I2 = Interval(R1('x'), 0.5, 1)
    I = I1 + I2
    I = I.boundary
    assert all(np.equal(I.bounding_box(), I1.bounding_box()))


def test_sample_random_uniform_in_union_boundary_with_n():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Circle(R2('x'), [0, 0], 1)
    P = (P1 + P2)
    points = P.boundary.sample_random_uniform(n=20)
    assert len(points) == 20
    assert torch.all(P.boundary._contains(points))


def test_sample_random_uniform_in_union_boundary_with_d():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Circle(R2('x'), [0, 0], 1)
    P = P1 + P2
    points = P.boundary.sample_random_uniform(d=12)
    assert torch.all(P.boundary._contains(points))


def test_sample_random_uniform_in_union_boundary_with_d_in_1D():
    I1 = Interval(R1('x'), 0, 1)
    I2 = Interval(R1('x'), 0.5, 1)
    I = I1 + I2
    points = I.boundary.sample_random_uniform(d=102)
    assert torch.all(I.boundary._contains(points))


def test_sample_grid_in_union_boundary_with_n():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Circle(R2('x'), [0.5, 0.5], 0.1)
    P = P1 + P2
    points = P.boundary.sample_grid(n=20)
    assert len(points) == 20
    assert torch.all(P.boundary._contains(points))


def test_sample_grid_in_union_boundary_with_d():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Circle(R2('x'), [0, 0], 1)
    P = P1 + P2
    points = P.boundary.sample_grid(d=12)
    assert torch.all(P.boundary._contains(points))


def test_sample_grid_in_union_boundary_with_d_in_1D():
    I1 = Interval(R1('x'), 0, 1)
    I2 = Interval(R1('x'), -0.3, 1)
    I = I1 + I2
    points = I.boundary.sample_grid(d=102)
    assert torch.all(I.boundary._contains(points))


def test_get_normals_of_union_in_2D():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Parallelogram(R2('x'), [-1, 0], [0, 0], [-1, 0.5])
    P = P1 + P2
    points = Points(torch.tensor([[0.7, 0.0], [1.0, 0.5], [0.3, 1], 
                                  [0.0, 0.8], [-0.2, 0.5], [-1, 0.2]]), R2('x'))
    normals = P.boundary.normal(points)
    expected_normals = torch.tensor([[0, -1.0], [1, 0], [0, 1], [-1, 0], 
                                     [0, 1], [-1, 0]])
    assert torch.allclose(normals, expected_normals)


def test_get_normals_of_union_in_1D():
    I1 = Interval(R1('x'), 0, 1)
    I2 = Interval(R1('x'), -3, 1)
    I = I1 + I2
    points = Points(torch.tensor([[-3.0], [1]]), R1('x'))
    normals = I.boundary.normal(points)
    expected_normals = torch.tensor([[-1], [1]])
    assert torch.allclose(normals, expected_normals)