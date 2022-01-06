import pytest
import torch
import numpy as np

from torchphysics.problem.domains.domain2D.circle import Circle
from torchphysics.problem.domains.domain2D.parallelogram import Parallelogram
from torchphysics.problem.domains.domain1D.interval import Interval
from torchphysics.problem.domains.domain3D.sphere import Sphere
from torchphysics.problem.spaces.space import R2, R1, R3
from torchphysics.problem.spaces.points import Points


def test_make_cut():
    C1 = Circle(R2('x'), [0, 0], 3)
    C2 = Circle(R2('x'), [0, 0], 1)
    C1 - C2


def test_cant_cut_from_different_spaces():
    C1 = Circle(R2('x'), [0, 0], 3)
    C2 = Circle(R2('y'), [0, 0], 1)
    with pytest.raises(ValueError):
        C1 - C2


def test_cut_register_nessecary_variables():
    C1 = Circle(R2('x'), [0, 0], lambda t: t+1)
    C2 = Circle(R2('x'), [0, 0], lambda u: u+3)
    C = C1 - C2
    assert 'u' in C.necessary_variables 
    assert 't' in C.necessary_variables 


def test_call_cut_domain():
    C1 = Circle(R2('x'), [0, 0], lambda t: t+1)
    C2 = Circle(R2('x'), [0, 0], 5)
    C = C2 - C1
    C(t=2)


def test_cut_contains_domain_constant_in_2D():
    C1 = Circle(R2('x'), [0, 0], 5)
    C2 = Circle(R2('x'), [0, 0], 2)
    C = C1 - C2  
    points = Points(torch.tensor([[0.0, 0.0], [0, -1], [0.5, -0.1],
                                  [4.1, 0], [-0.1, -3]]), R2('x'))
    inside = C._contains(points)
    assert not any(inside[:3])
    assert all(inside[3:])


def test_cut_contains_domain_changes_in_2D():
    C1 = Circle(R2('x'), [0, 0], 5)
    C2 = Circle(R2('x'), [0, 0], lambda t: t+1)
    C = C1 - C2  
    points = Points(torch.tensor([[0.0, 0.0], [0, -1], [0.5, -0.1],
                                  [3.1, 0], [-0.1, -3], [0.0, 0.0], [0, -1],
                                  [0.5, -0.1], [3.1, 0], [-0.1, 4.5]]), R2('x'))
    time = Points(torch.tensor([[1.0], [3.0]]).repeat_interleave(5, dim=0), R1('t'))
    inside = C._contains(points, time)
    assert not any(inside[:3])
    assert all(inside[3:5])
    assert not any(inside[5:9])
    assert inside[9]


def test_cut_contains_domain_in_1D():
    C1 = Interval(R1('x'), 0, 1)
    C2 = Interval(R1('x'), 0.5, 1)
    C = C1 - C2  
    points = Points(torch.tensor([[0.0], [0.45], [0.5001],
                                  [-10], [0.8]]), R1('x'))
    inside = C._contains(points)
    assert not any(inside[2:])
    assert all(inside[:2])


def test_cut_contains_domain_in_3D():
    C1 = Sphere(R3('x'), [0, 0, 0], 3)
    C2 = Sphere(R3('x'), [0, 0, 0], 1)
    C = C1 - C2  
    points = Points(torch.tensor([[0.0, 2.0, 0.0], [1.5, 0.2, 0], [0.5001, 0, 0],
                                  [-10, 0, 0], [0.8, 0, 0.03]]), R3('x'))
    inside = C._contains(points)
    assert not any(inside[2:])
    assert all(inside[:2])


def test_get_volume_of_cut_if_contained():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Parallelogram(R2('x'), [0, 0], [0.5, 0], [0, 0.5])
    P = P1 - P2
    P.contained = True
    assert torch.isclose(P._get_volume(), torch.tensor(0.75))


def test_get_volume_of_cut_if_not_contained():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Parallelogram(R2('x'), [-0.5, -0.5], [0.5, 0], [0, 0.5])
    P = P1 - P2
    with pytest.warns(UserWarning):
        volume = P._get_volume()
    assert torch.isclose(volume, torch.tensor(1.0))


def test_get_volume_of_cut_if_domain_changes():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    C2 = Circle(R2('x'), [0.5, 0.5], lambda t: 0.1*t)
    P = P1 - C2
    P.contained = True
    time = Points(torch.tensor([[1.0], [2.0]]), R1('t'))
    assert torch.allclose(P._get_volume(time),
                          torch.tensor([[1 - np.pi/100], [1 - np.pi/25]]))


def test_bounding_box_of_cut():
    I1 = Interval(R1('x'), 0, 1)
    I2 = Interval(R1('x'), 0.5, 1)
    I = I1 - I2
    assert all(np.equal(I.bounding_box(), I1.bounding_box()))


def test_sample_random_uniform_in_cut_with_n():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Parallelogram(R2('x'), [-0.5, -0.5], [0.5, 0], [0, 0.5])
    P = P1 - P2
    points = P.sample_random_uniform(n=100)
    assert len(points) == 100
    assert torch.all(P._contains(points))


def test_sample_random_uniform_in_cut_with_n_and_product():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Parallelogram(R2('x'), [-0.5, -0.5], [0.5, 0], [0, 0.5])
    I = Interval(R1('t'), 0, 1)
    P = (P1 - P2) * I
    points = P.sample_random_uniform(n=100)
    assert len(points) == 100
    assert torch.all(P._contains(points))


def test_sampler_random_uniform_in_cut_with_n_and_nothing_cut():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Parallelogram(R2('x'), [-0.5, -0.5], [0, -0.5], [-0.5, -0.2])
    P = P1 - P2
    points = P.sample_random_uniform(n=100)
    assert len(points) == 100
    assert torch.all(P._contains(points))


def test_sample_random_uniform_in_cut_with_d():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Parallelogram(R2('x'), [0, 0], [0.5, 0], [0, 1])
    P = P1 - P2
    points = P.sample_random_uniform(d=20)
    assert torch.all(points.as_tensor[:, 0] >= 0.5)
    assert torch.all(P1._contains(points))


def test_sample_random_uniform_in_cut_with_d_in_1D():
    I1 = Interval(R1('x'), 0, 1)
    I2 = Interval(R1('x'), 0.5, 1)
    I = I1 - I2
    points = I.sample_random_uniform(d=20)
    assert torch.all(points.as_tensor <= 0.5)
    assert torch.all(I1._contains(points))


def test_sample_grid_in_cut_with_n():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Parallelogram(R2('x'), [-0.5, -0.5], [0.5, 0], [0, 0.5])
    P = P1 - P2
    points = P.sample_grid(n=100)
    assert len(points) == 100
    assert torch.all(P._contains(points))


def test_sample_grid_in_cut_with_n_and_nothing_cut():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Parallelogram(R2('x'), [-0.5, -0.5], [0, -0.5], [-0.5, -0.2])
    P = P1 - P2
    points = P.sample_grid(n=100)
    assert len(points) == 100
    assert torch.all(P._contains(points))


def test_sample_grid_in_cut_with_d():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Parallelogram(R2('x'), [0, 0], [0.5, 0], [0, 1])
    P = P1 - P2
    points = P.sample_grid(d=202)
    assert torch.all(points.as_tensor[:, 0] >= 0.5)
    assert torch.all(P1._contains(points))


def test_sample_grid_in_cut_with_d_in_1D():
    I1 = Interval(R1('x'), 0, 1)
    I2 = Interval(R1('x'), 0.3, 1)
    I = I1 - I2
    points = I.sample_grid(d=20)
    assert torch.all(points.as_tensor <= 0.5)
    assert torch.all(I1._contains(points))


def test_get_cut_boundary():
    C1 = Circle(R2('x'), [0, 0], 3)
    C2 = Circle(R2('x'), [0, 0], 1)
    C = C1 - C2
    C.boundary


def test_cut_boundary_contains_domain_constant_in_2D():
    C1 = Circle(R2('x'), [0, 0], 5)
    C2 = Circle(R2('x'), [0, 0], 2)
    C = C1 - C2 
    C = C.boundary 
    points = Points(torch.tensor([[0.0, 0.0], [0, -1], [0.5, -0.1],
                                  [5, 0], [2, 0], [0, -2], [-5, 0]]), R2('x'))
    inside = C._contains(points)
    assert not any(inside[:3])
    assert all(inside[3:])


def test_cut_boundary_contains_domain_changes_in_2D():
    C1 = Circle(R2('x'), [0, 0], 5)
    C2 = Circle(R2('x'), [0, 0], lambda t: t+1)
    C = C1 - C2  
    C = C.boundary 
    points = Points(torch.tensor([[0.0, 0.0], [0, -1], [0.5, -0.1],
                                  [5, 0], [0, -2], [0.0, 2], [0, -1],
                                  [0.5, -0.1], [4, 0], [0, -5]]), R2('x'))
    time = Points(torch.tensor([[1.0], [3.0]]).repeat_interleave(5, dim=0), R1('t'))
    inside = C._contains(points, time)
    assert not any(inside[:3])
    assert all(inside[3:5])
    assert not any(inside[5:8])
    assert all(inside[8:])


def test_cut_boundary_contains_domain_in_1D():
    C1 = Interval(R1('x'), 0, 1)
    C2 = Interval(R1('x'), 0.5, 1)
    C = C1 - C2  
    C = C.boundary 
    points = Points(torch.tensor([[1.0], [-1], [0.5001],
                                  [0.5], [0.0]]), R1('x'))
    inside = C._contains(points)
    assert not any(inside[:3])
    assert all(inside[3:])


def test_cut_boundary_contains_domain_in_3D():
    C1 = Sphere(R3('x'), [0, 0, 0], 3)
    C2 = Sphere(R3('x'), [0, 0, 0], 1)
    C = C1 - C2  
    C = C.boundary 
    points = Points(torch.tensor([[0.0, 3.0, 0.0], [-1, 0, 0], [0, 1, 0],
                                  [3, 0, 0], [0.8, 0, 0.03]]), R3('x'))
    inside = C._contains(points)
    assert not any(inside[4:])
    assert all(inside[:4])


def test_get_volume_of_cut_boundary_if_contained():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    C2 = Circle(R2('x'), [0.5, 0.5], 0.2)
    P = P1 - C2
    P = P.boundary 
    P.contained = True
    assert torch.isclose(P._get_volume(), torch.tensor(4+0.4*np.pi))


def test_get_volume_of_cut_boundary_if_not_contained():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Parallelogram(R2('x'), [0, 0], [0.5, 0], [0, 0.5])
    P = P1 - P2
    P = P.boundary
    with pytest.warns(UserWarning):
        volume = P._get_volume()
    assert torch.isclose(volume, torch.tensor(6.0))


def test_get_volume_of_cut_boundary_if_domain_changes():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    C2 = Circle(R2('x'), [0.5, 0.5], lambda t: 0.1*t)
    P = P1 - C2
    P.contained = True
    P = P.boundary
    time = Points(torch.tensor([[1.0], [2.0]]), R1('t'))
    assert torch.allclose(P._get_volume(time),
                          torch.tensor([[4 + 0.2*np.pi], [4 + 0.4*np.pi]]))


def test_bounding_box_of_cut_boundary():
    I1 = Interval(R1('x'), 0, 1)
    I2 = Interval(R1('x'), 0.5, 1)
    I = I1 - I2
    I = I.boundary
    assert all(np.equal(I.bounding_box(), I1.bounding_box()))


def test_sample_random_uniform_in_cut_boundary_with_n():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Parallelogram(R2('x'), [-0.5, -0.5], [0.5, 0], [0, 0.5])
    P = (P1 - P2)
    points = P.boundary.sample_random_uniform(n=25)
    assert len(points) == 25
    assert torch.all(P.boundary._contains(points))


def test_sample_random_uniform_in_cut_boundary_with_n_and_product():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Parallelogram(R2('x'), [-0.5, -0.5], [0.5, 0], [0, 0.5])
    I = Interval(R1('t'), 0, 1)
    P = (P1 - P2) * I
    points = P.boundary.sample_random_uniform(n=25)
    assert len(points) == 25
    assert torch.all(P.boundary._contains(points))


def test_sample_random_in_cut_boundary_with_n_and_params():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Circle(R2('x'), [0, 0], 0.5)
    P = P1 - P2
    time = Points(torch.tensor([[1.0], [0.3]]), R1('t'))
    points = P.boundary.sample_random_uniform(n=25, params=time)
    assert len(points) == 50
    assert torch.all(P.boundary._contains(points))

def test_sample_random_uniform_in_cut_boundary_with_d():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Parallelogram(R2('x'), [0, 0], [0.5, 0], [0, 1])
    P = P1 - P2
    points = P.boundary.sample_random_uniform(d=20)
    assert torch.all(points.as_tensor[:, 0] >= 0.5)
    assert torch.all(P.boundary._contains(points))


def test_sample_random_uniform_in_cut_boundary_with_d_in_1D():
    I1 = Interval(R1('x'), 0, 1)
    I2 = Interval(R1('x'), 0.5, 1)
    I = I1 - I2
    points = I.boundary.sample_random_uniform(d=20)
    assert torch.all(points.as_tensor <= 0.5)
    assert torch.all(I.boundary._contains(points))


def test_sample_grid_in_cut_boundary_with_n():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Circle(R2('x'), [0, 0], 0.5)
    P = P1 - P2
    points = P.boundary.sample_grid(n=25)
    assert len(points) == 25
    assert torch.all(P.boundary._contains(points))


def test_sample_grid_in_cut_boundary_with_d():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Parallelogram(R2('x'), [0, 0], [0.5, 0], [0, 1])
    P = P1 - P2
    points = P.boundary.sample_grid(d=20)
    assert torch.all(points.as_tensor[:, 0] >= 0.5)
    assert torch.all(P.boundary._contains(points))


def test_sample_grid_in_cut_boundary_with_d_in_1D():
    I1 = Interval(R1('x'), 0, 1)
    I2 = Interval(R1('x'), 0.3, 1)
    I = I1 - I2
    points = I.boundary.sample_grid(d=20)
    assert torch.all(points.as_tensor <= 0.3)
    assert torch.all(I.boundary._contains(points))


def test_get_normals_of_cut_in_2D():
    P1 = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    P2 = Parallelogram(R2('x'), [0, 0], [0.5, 0], [0, 0.5])
    P = P1 - P2
    points = Points(torch.tensor([[0.7, 0.0], [1.0, 0.5], [0.3, 1], 
                                  [0.0, 0.8], [0.2, 0.5], [0.5, 0.2]]), R2('x'))
    normals = P.boundary.normal(points)
    expected_normals = torch.tensor([[0, -1.0], [1, 0], [0, 1], [-1, 0], 
                                     [0, -1], [-1, 0]])
    assert torch.allclose(normals, expected_normals)


def test_get_normals_of_cut_in_1D():
    I1 = Interval(R1('x'), 0, 1)
    I2 = Interval(R1('x'), 0.3, 1)
    I = I1 - I2
    points = Points(torch.tensor([[0.0], [0.3]]), R1('x'))
    normals = I.boundary.normal(points)
    expected_normals = torch.tensor([[-1], [1]])
    assert torch.allclose(normals, expected_normals)