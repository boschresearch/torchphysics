import pytest
import torch
import numpy as np

from torchphysics.problem.domains.domain2D.circle import Circle
from torchphysics.problem.domains.domain2D.parallelogram import Parallelogram
from torchphysics.problem.domains.domain1D.interval import Interval
from torchphysics.problem.domains.domain3D.sphere import Sphere
from torchphysics.problem.domains.domainoperations.translate import Translate
from torchphysics.problem.domains.domainoperations.rotate import Rotate
from torchphysics.problem.spaces.space import R2, R1
from torchphysics.problem.spaces.points import Points

########################################
# Translation tests

def trans_fn(x):
    return x


def test_create_translation():
    T = R1('t')
    I = Interval(T, 0, 1.0)
    I_shift = Translate(I, trans_fn)
    assert I_shift.domain == I
    assert I_shift.dim == 1
    assert I_shift.space == T


def test_call_translation():
    T = R1('t')
    I = Interval(T, 0, 1.0)
    I_shift = Translate(I, trans_fn)
    I_shift = I_shift(t=2.0)
    assert isinstance(I_shift, Translate)


def test_boundary_of_translation():
    T = R1('t')
    I = Interval(T, 0, 1.0)
    I_shift = Translate(I, trans_fn)
    I_shift_boundary = I_shift.boundary
    assert isinstance(I_shift_boundary, Translate)
    assert I_shift_boundary.domain.domain == I


def test_get_volume_of_translation():
    T = R1('t')
    I = Interval(T, 0, 1.0)
    I_shift = Translate(I, trans_fn)
    assert I_shift.volume() == 1.0
    I_shift.set_volume(2.0)
    assert I_shift.volume() == 2.0


def test_translation_contains_1D():
    T, X = R1('t'), R1('x')
    I = Interval(T, 0, 1.0)
    I_shift = Translate(I, trans_fn)
    points = Points(torch.tensor([[0.0, 0.0], [1.0, 0.5], [1.2, 1.0], 
                                  [0.5, 1.0], [3.5, 1.5]]), T*X)
    in_values = I_shift._contains(points[:, 't'], points[:, 'x'])
    assert all(in_values == torch.tensor([True, True, True, False, False]).reshape(-1, 1))


def test_constant_translation_contains_1D():
    T = R1('t')
    I = Interval(T, 0, 1.0)
    I_shift = Translate(I, 0.5)
    points = Points(torch.tensor([[0.0], [1.0], [1.2], [0.5], [3.5]]), T)
    in_values = I_shift._contains(points)
    assert all(in_values == torch.tensor([False, True, True, True, False]).reshape(-1, 1))


def test_translation_contains_2D():
    T, X = R2('t'), R2('x')
    I = Circle(T, [0, 0], 2.0)
    I_shift = Translate(I, trans_fn)
    points = Points(torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 0.5, 0.0, 0.0], 
                                  [3.2, 1.0, 0.0, 0.0], [3.2, 1.0, 2.0, 0.0], 
                                  [3.5, -1.0, 7.0, 0.0], [0.5, -3.5, 0.0, -2.0]]), T*X)
    in_values = I_shift._contains(points[:, 't'], points[:, 'x'])
    assert all(in_values == torch.tensor([True, True, False, True, False, True]).reshape(-1, 1))


def test_constant_translation_contains_2D():
    T = R2('t')
    I = Circle(T, [0, 0], 2.0)
    I_shift = Translate(I, [2.0, 0.0])
    points = Points(torch.tensor([[-1.0, 0.0], [1.0, 0.0], [1.2, 1.0], 
                                  [3.5, -1.0], [3.5, 4.0]]), T)
    in_values = I_shift._contains(points)
    assert all(in_values == torch.tensor([False, True, True, True, False]).reshape(-1, 1))


def test_translation_contains_double_dependent():
    T, X = R2('t'), R2('x')
    def r(x):
        return x[:, :1] + 1
    I = Circle(T, [0, 0], r)
    I_shift = Translate(I, trans_fn)
    points = Points(torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 0.5, 0.0, 0.0], 
                                  [3.2, 1.0, 0.0, 0.0], [3.2, 1.0, 2.0, 0.0], 
                                  [3.5, -1.0, 7.0, 0.0], [0.5, -3.5, 0.0, -2.0]]), T*X)
    in_values = I_shift._contains(points[:, 't'], points[:, 'x'])
    assert all(in_values == torch.tensor([True, False, False, True, True, False]).reshape(-1, 1))


def test_random_sample_in_translation():
    T = R2('t')
    I = Circle(T, [0, 0], 2.0)
    I_shift = Translate(I, [2.0, 0.0])
    p = I_shift.sample_random_uniform(n=300)
    assert p.as_tensor.shape == (300, 2) 
    assert all(I_shift._contains(p))


def test_random_sample_in_translation_with_params():
    T = R1('t')
    I = Interval(T, 0, 1.0)
    I_shift = Translate(I, trans_fn)
    params = Points(torch.linspace(0, 1, 300).reshape(-1, 1), R1('x'))
    p = I_shift.sample_random_uniform(n=1, params=params)
    assert p.as_tensor.shape == (300, 1) 
    assert all(I_shift._contains(p, params=params))


def test_random_sample_in_translation_with_params_2():
    T = R1('t')
    I = Interval(T, -1, trans_fn)
    I_shift = Translate(I, trans_fn)
    params = Points(torch.linspace(0, 1, 5).reshape(-1, 1), R1('x'))
    p = I_shift.sample_random_uniform(n=5, params=params)
    params_repeat = Points(torch.repeat_interleave(params, 5, dim=0), R1('x'))
    assert p.as_tensor.shape == (25, 1) 
    assert all(I_shift._contains(p, params=params_repeat))


def test_grid_sample_in_translation():
    T = R2('t')
    I = Circle(T, [0, 0], 2.0)
    I_shift = Translate(I, [2.0, 0.0])
    p = I_shift.sample_grid(n=30)
    assert p.as_tensor.shape == (30, 2) 
    assert all(I_shift._contains(p))


def test_grid_sample_in_translation_with_params():
    T = R1('t')
    I = Interval(T, 0, 1.0)
    I_shift = Translate(I, trans_fn)
    params = Points(torch.linspace(0, 1, 300).reshape(-1, 1), R1('x'))
    p = I_shift.sample_grid(n=1, params=params)
    assert p.as_tensor.shape == (300, 1) 
    assert all(I_shift._contains(p, params=params))


def test_grid_sample_in_translation_with_params_2():
    T = R1('t')
    I = Interval(T, -1, trans_fn)
    I_shift = Translate(I, trans_fn)
    params = Points(torch.linspace(0, 1, 5).reshape(-1, 1), R1('x'))
    p = I_shift.sample_grid(n=5, params=params)
    params_repeat = Points(torch.repeat_interleave(params, 5, dim=0), R1('x'))
    assert p.as_tensor.shape == (25, 1) 
    assert all(I_shift._contains(p, params=params_repeat))


def test_bounds_of_translation():
    T = R1('t')
    I = Interval(T, 0, 1.0)
    I_shift = Translate(I, 2.0)
    assert torch.all(I_shift.bounding_box() == torch.tensor([[2.0, 3.0]]))


def test_bounds_of_translation_2D():
    T = R2('t')
    I = Circle(T, [0, 0], 2.0)
    I_shift = Translate(I, [2.0, 0.0])
    assert torch.all(I_shift.bounding_box() == torch.tensor([[0.0, 4.0, -2.0, 2.0]]))


def test_bounds_of_translation_with_params():
    T = R1('t')
    I = Interval(T, 0, 1.0)
    I_shift = Translate(I, trans_fn)
    params = Points(torch.tensor([[0.0], [1.0], [2.0]]), R1('x'))
    bounds = I_shift.bounding_box(params)
    real_bounds = torch.tensor([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    assert torch.all(bounds == real_bounds)

##########################################
# rotation 

def rotate_matrix_fn(t):
    matrix_row = torch.cat((torch.cos(np.pi * t), -torch.sin(np.pi * t)), dim=1)
    matrix_row_2 = torch.cat((-matrix_row[:, 1:], matrix_row[:, :1]), dim=1)
    return torch.stack((matrix_row, matrix_row_2), dim=1)

def angle_fn(t):
    return np.pi * t


def test_create_rotation():
    X = R2('x')
    box = Parallelogram(X, [0, 0], [1, 0], [0, 1])
    box_rot = Rotate(box, torch.tensor([[-1, 0], [0, -1]]))
    assert box_rot.domain == box
    assert box_rot.dim == 2
    assert box_rot.space == X


def test_create_rotation_with_fn():
    X = R2('x')
    box = Parallelogram(X, [0, 0], [1, 0], [0, 1])
    box_rot = Rotate(box, rotate_matrix_fn, rotate_around=[0, 1])
    assert box_rot.domain == box
    assert box_rot.dim == 2
    assert box_rot.space == X


def test_create_rotation_with_angle():
    X = R2('x')
    box = Parallelogram(X, [0, 0], [1, 0], [0, 1])
    box_rot = Rotate.from_angles(box, 2.5)
    assert box_rot.domain == box
    assert box_rot.dim == 2
    assert box_rot.space == X


def test_create_rotation_in_3D():
    X = R2('x') * R1('z')
    box = Sphere(X, [0, 0], 1.0)
    with pytest.raises(NotImplementedError):
        _ = Rotate.from_angles(box, 2.5, 0.0, 2.0)


def test_call_rotation():
    X = R2('x')
    box = Parallelogram(X, [0, 0], [1, 0], [0, 1])
    box_rot = Rotate.from_angles(box, angle_fn)
    box_rot = box_rot(t=2.0)
    assert isinstance(box_rot, Rotate)


def test_boundary_of_rotation():
    X = R2('x')
    box = Parallelogram(X, [0, 0], [1, 0], [0, 1])
    box_rot = Rotate.from_angles(box, angle_fn)
    box_rot_boundary = box_rot.boundary
    assert isinstance(box_rot_boundary, Rotate)
    assert box_rot_boundary.domain.domain == box


def test_get_volume_of_rotation():
    X = R2('x')
    box = Parallelogram(X, [0, 0], [1, 0], [0, 1])
    box_rot = Rotate.from_angles(box, angle_fn)
    assert box_rot.volume() == 1.0
    box_rot.set_volume(2.0)
    assert box_rot.volume() == 2.0


def test_rotation_contains_with_angle():
    X = R2('x')
    box = Parallelogram(X, [0, 0], [1, 0], [0, 1])
    box_rot = Rotate.from_angles(box, np.pi/2.0)
    points = Points(torch.tensor([[-0.5, 0.5], [-0.7, 0.1], [-0.2, 0.98], 
                                  [0.5, 0.5], [-0.5, -1.0]]), X)
    in_values = box_rot._contains(points)
    assert all(in_values == torch.tensor([True, True, True, False, False]).reshape(-1, 1))


def test_rotation_contains_with_angle_dependent():
    X, T = R2('x'), R1('t')
    box = Parallelogram(X, [0, 0], [1, 0], [0, 1])
    box_rot = Rotate.from_angles(box, angle_fn)
    points = Points(torch.tensor([[-0.5, 0.5, 0.5], [-0.7, 0.1, 0.5], [-0.2, 0.98, 0.0], 
                                  [0.5, 0.5, 0.0], [-0.5, -1.0, 1.0], [-0.2, 0.98, 1.0],
                                  [0.5, 0.5, 1.5], [0.5, -1.0, 1.5]]), X*T)
    in_values = box_rot._contains(points)
    assert all(in_values == torch.tensor([True, True, False, True, 
                                          True, False, False, True]).reshape(-1, 1))


def test_rotation_contains_with_matrix_dependent():
    X, T = R2('x'), R1('t')
    box = Parallelogram(X, [0, 0], [1, 0], [0, 1])
    box_rot = Rotate(box, rotate_matrix_fn)
    points = Points(torch.tensor([[-0.5, 0.5, 0.5], [-0.7, 0.1, 0.5], [-0.2, 0.98, 0.0], 
                                  [0.5, 0.5, 0.0], [-0.5, -1.0, 1.0], [-0.2, 0.98, 1.0],
                                  [0.5, 0.5, 1.5], [0.5, -1.0, 1.5]]), X*T)
    in_values = box_rot._contains(points)
    assert all(in_values == torch.tensor([True, True, False, True, 
                                          True, False, False, True]).reshape(-1, 1))


def test_rotation_contains_with_rotate_around():
    X = R2('x')
    box = Parallelogram(X, [0, 0], [1, 0], [0, 1])
    box_rot = Rotate.from_angles(box, np.pi/2.0, rotate_around=[1.0, 1.0])
    points = Points(torch.tensor([[-0.5, 0.5], [-0.7, 0.1], [-0.2, 0.98], 
                                  [0.5, 0.5], [-0.5, -1.0], [1.2, 0.98],
                                  [0.5, -0.5], [1.9, 0.2], [1.2, 1.08]]), X)
    in_values = box_rot._contains(points)
    assert all(in_values == torch.tensor([False, False, False, False, False, 
                                          True, False, True, False]).reshape(-1, 1))


def test_random_sample_in_rotation():
    X, T = R2('x'), R1('t')
    box = Parallelogram(X, [0, 0], [1, 0], [0, 1])
    box_rot = Rotate.from_angles(box, angle_fn)
    params = Points(torch.linspace(0, 1, 300).reshape(-1, 1), T)
    p = box_rot.sample_random_uniform(n=1, params=params)
    assert p.as_tensor.shape == (300, 2) 
    assert all(box_rot._contains(p, params=params))


def test_grid_sample_in_rotation():
    X = R2('x')
    box = Parallelogram(X, [0, 0], [1, 0], [0, 1])
    box_rot = Rotate.from_angles(box, 2.0, rotate_around=[2.0, 1.0])
    p = box_rot.sample_grid(n=30)
    assert p.as_tensor.shape == (30, 2) 
    assert all(box_rot._contains(p))


def test_grid_sample_in_rotation_with_params():
    X = R2('x')
    box = Parallelogram(X, [0, 0], [1, 0], [0, 1])
    box_rot = Rotate.from_angles(box, angle_fn)
    params = Points(torch.linspace(0, 1, 300).reshape(-1, 1), R1('t'))
    p = box_rot.sample_grid(n=1, params=params)
    assert p.as_tensor.shape == (300, 2) 
    assert all(box_rot._contains(p, params=params))


def test_grid_sample_in_rotation_with_params_2():
    def corner_1(t):
        corner = torch.zeros((len(t), 2))
        corner[:, :1] = -t
        return corner
    X = R2('x')
    box = Parallelogram(X, corner_1, [1, 0], [0, 1])
    box_rot = Rotate.from_angles(box, angle_fn)
    params = Points(torch.tensor([[1.0]]), R1('t'))
    p = box_rot.sample_grid(n=5, params=params)
    params_repeat = Points(torch.repeat_interleave(params, 5, dim=0), R1('t'))
    assert p.as_tensor.shape == (5, 2) 
    assert all(box_rot._contains(p, params=params_repeat))


def test_bounds_of_rotation():
    X = R2('x')
    box = Parallelogram(X, [0, 0], [1, 0], [0, 1])
    box_rot = Rotate.from_angles(box, np.pi/2.0, rotate_around=[1.0, 1.0])
    assert torch.allclose(box_rot.bounding_box(), 
                          torch.tensor([[1.0, 2.0, 0.0, 1.0]]), 
                          atol=1e-6)