import torch
import pytest

from torchphysics.problem.spaces import R1, R2, Points


def test_create_points():
    p = Points(torch.ones((3, 1)), R1('x'))
    assert p.space.dim == 1
    assert isinstance(p._t, torch.Tensor)


def test_get_tensor_from_point():
    p = Points(torch.ones((3, 1)), R1('x'))
    assert isinstance(p.as_tensor, torch.Tensor) 
    assert torch.equal(p.as_tensor, torch.ones((3, 1)))


def test_length_of_points():
    p = Points(torch.ones((33, 2)), R2('x'))
    assert len(p) == 33 


def test_create_points_space_has_to_fit_points():
    with pytest.raises(AssertionError):
        Points(torch.ones((3, 2)), R1('x'))


def test_create_points_data_has_to_have_batch_dim():
    with pytest.raises(AssertionError):
        Points(torch.ones(3), R1('x'))


def test_create_empty_points():
    p = Points.empty()
    assert p.space.dim == 0


def test_join_points():
    p1 = Points(torch.ones((2, 1)), R1('x'))
    p2 = Points(torch.zeros((2, 2)), R2('y'))
    p = Points.joined(p1, p2)
    assert p.space.dim == 3
    assert torch.equal(p._t, torch.tensor([[1.0, 0, 0], [1, 0, 0]]))


def test_join_points_with_empty_point():
    p1 = Points(torch.ones((2, 1)), R1('x'))
    p2 = Points.empty()
    p = Points.joined(p1, p2)
    assert p.space.dim == 1
    assert torch.equal(p._t, torch.tensor([[1.0], [1]]))


def test_create_points_from_dict():
    inp_dict = {'x': torch.ones((2, 1)), 'y': 2*torch.ones((2, 2))}
    p = Points.from_coordinates(inp_dict)
    assert p.space.dim == 3
    assert 'x' in p.space
    assert 'y' in p.space
    assert torch.equal(p._t, torch.tensor([[1.0, 2, 2], [1, 2, 2]]))


def test_create_points_from_empty_dict():
    p = Points.from_coordinates({})
    assert p.space.dim == 0


def test_check_dim_of_points():
    p = Points(torch.ones((3, 1)), R1('x'))
    assert p.dim == 1


def test_check_variables_of_points():
    r = R1('x')
    p = Points(torch.ones((3, 1)), r)
    assert p.variables == r.variables


def test_points_to_coordinate_dict():
    p = Points(torch.ones((3, 3)), R1('x')*R2('u'))
    coords = p.coordinates
    assert isinstance(coords, dict)
    assert 'x' in coords.keys()
    assert 'u' in coords.keys()
    assert torch.equal(coords['x'], torch.ones((3, 1)))
    assert torch.equal(coords['u'], torch.ones((3, 2)))


def test_check_points_empty():
    p = Points(torch.ones((3, 3)), R1('x')*R2('u'))
    assert not p.isempty
    p = Points.empty()
    assert p.isempty 


def test_check_points_equal():
    p1 = Points(torch.ones((3, 1)), R1('x'))
    p2 = Points(torch.ones((3, 1)), R1('y'))
    p3 = Points(torch.ones((2, 1)), R1('x'))
    assert p1 == p1
    assert not p1 == p2
    assert not p1 == p3


def test_points_get_item():
    X = R2('x')
    T = R1('t')
    p = Points(torch.tensor([[2.0,1.0,2.5], [1.0,3.0,1.0],
                             [1.0,1.0,4.0], [1.0,5.0,1.0],
                             [6.0,1.0,1.0]]), X*T)

    assert torch.all(p[1,'x'].as_tensor == torch.tensor([1.0, 3.0]))
    assert torch.all(p[1:3,'x'].as_tensor == torch.tensor([[1.0, 3.0],
                                                           [1.0, 1.0]]))
    assert p[2:5,:] == p[2:5]
    assert p[2:5,:] == Points(torch.tensor([[1., 1., 4.], [1., 5., 1.],
                                            [6., 1., 1.]]), X*T)
    expected_coords = {'x': torch.tensor([[1., 1.], [1., 5.], [6., 1.]]),
                       't': torch.tensor([[4.], [1.], [1.]])}
    computed_coords = p[2:5,:].coordinates 
    assert torch.equal(computed_coords['x'], expected_coords['x'])
    assert torch.equal(computed_coords['t'], expected_coords['t'])


def test_points_tensor_slice():
    p = Points(torch.tensor([[1, 2], [3.0, 4.0]]), R1('x')*R1('t'))
    slc = (p.as_tensor > 2)
    assert p[slc[:,0]] == Points(torch.tensor([[3.0, 4.0]]), R1('x')*R1('t'))
    with pytest.raises(IndexError):
        p[slc]


def test_iterate_over_points():
    p = Points(torch.tensor([[2], [2.0]]), R1('x'))
    iter_p = iter(p)
    for p in iter_p:
        assert isinstance(p, Points)
        assert p.as_tensor == 2.0


def test_add_points():
    p1 = Points(torch.tensor([[2], [1.0]]), R1('x'))
    p = p1 + p1
    assert isinstance(p, Points)
    assert torch.equal(torch.tensor([[4.0], [2.0]]), p.as_tensor)


def test_cant_add_points_of_different_spaces():
    p1 = Points(torch.tensor([[2], [1.0]]), R1('x'))
    p2 = Points(torch.tensor([[2], [1.0]]), R1('y'))
    with pytest.raises(AssertionError):
        p1 + p2


def test_substract_points():
    p1 = Points(torch.tensor([[2, 1], [1.0, 0]]), R2('x'))
    p = p1 - p1
    assert isinstance(p, Points)
    assert torch.equal(torch.tensor([[0.0, 0], [0.0, 0]]), p.as_tensor)


def test_cant_substract_points_of_different_spaces():
    p1 = Points(torch.tensor([[2], [1.0]]), R1('x'))
    p2 = Points(torch.tensor([[2], [1.0]]), R1('y'))
    with pytest.raises(AssertionError):
        p1 + p2


def test_multiply_points():
    p1 = Points(torch.tensor([[2], [1.0]]), R1('x'))
    p = p1 * p1
    assert isinstance(p, Points)
    assert torch.equal(torch.tensor([[4.0], [1.0]]), p.as_tensor)


def test_cant_multiply_points_of_different_spaces():
    p1 = Points(torch.tensor([[2], [1.0]]), R1('x'))
    p2 = Points(torch.tensor([[2], [1.0]]), R1('y'))
    with pytest.raises(AssertionError):
        p1 * p2


def test_divide_points():
    p1 = Points(torch.tensor([[2], [1.0]]), R1('x'))
    p = p1 / p1
    assert isinstance(p, Points)
    assert torch.equal(torch.tensor([[1.0], [1.0]]), p.as_tensor)


def test_cant_divide_points_of_different_spaces():
    p1 = Points(torch.tensor([[2], [1.0]]), R1('x'))
    p2 = Points(torch.tensor([[2], [1.0]]), R1('y'))
    with pytest.raises(AssertionError):
        p1 / p2


def test_raise_power_points():
    p1 = Points(torch.tensor([[2], [1.0]]), R1('x'))
    p = p1**p1
    assert isinstance(p, Points)
    assert torch.equal(torch.tensor([[4.0], [1.0]]), p.as_tensor)


def test_cant_raise_power_points_of_different_spaces():
    p1 = Points(torch.tensor([[2], [1.0]]), R1('x'))
    p2 = Points(torch.tensor([[2], [1.0]]), R1('y'))
    with pytest.raises(AssertionError):
        p1**p2


def test_or_for_points():
    p1 = Points(torch.tensor([[2], [1.0]]), R1('x'))
    p2 = Points(torch.tensor([[2], [1.0]]), R1('x'))
    p = p1 | p2
    assert isinstance(p, Points)
    assert 'x' in p.space
    assert len(p) == 4


def test_or_for_empty_points():
    p1 = Points(torch.tensor([[2], [1.0]]), R1('x'))
    p2 = Points.empty()
    p = p1 | p2
    assert p == p1
    p = p2 | p1
    assert p == p1


def test_join_points():
    p1 = Points(torch.tensor([[2], [1.0]]), R1('x'))
    p2 = Points(torch.tensor([[2], [1.0]]), R1('y'))
    p = p1.join(p2)
    assert isinstance(p, Points)
    assert 'x' in p.space
    assert 'y' in p.space
    assert len(p) == 2


def test_points_setitem():
    p1 = Points(torch.tensor([[2, 3], [1.0, 4]]), R1('x')*R1('y'))
    p2 = Points(torch.tensor([[2], [1.0]]), R1('y'))
    p1[..., 'y'] = p2
    assert p1 == Points(torch.tensor([[2, 2], [1.0, 1]]), R1('x')*R1('y'))


def test_join_empty_points():
    p1 = Points(torch.tensor([[2], [1.0]]), R1('x'))
    p2 = Points.empty()
    p = p1.join(p2)
    assert p == p1
    p = p2.join(p1)
    assert p == p1


def test_repeat_points():
    p1 = Points(torch.tensor([[2], [1.0]]), R1('x'))
    p = p1.repeat(4)
    assert isinstance(p, Points)
    assert len(p) == 8
    assert 'x' in p.space
    assert p._t[0] == p._t[2]
    assert not p._t[0] == p._t[1]
    assert p._t[5] == p._t[7]


def test_requires_grad_points():
    p = Points(torch.tensor([[2], [1.0]]), R1('x'))
    assert not p.requires_grad


def test_set_requires_grad_points():
    p = Points(torch.tensor([[2], [1.0]]), R1('x'))
    assert not p.requires_grad
    p.requires_grad = True
    assert p.requires_grad


def test_torch_function_for_points():
    p = Points(torch.tensor([[2], [1.0]]), R1('x'))
    p_repeat = torch.repeat_interleave(p, 10, dim=0)
    assert isinstance(p_repeat, torch.Tensor)
    assert len(p_repeat) == 20
    assert p_repeat[0] == p_repeat[1]
    assert p_repeat[10] == p_repeat[11]


def test_torch_function_for_points_2():
    p = Points(torch.tensor([[2], [1.0]]), R1('x'))
    p_repeat = torch.neg(p)
    assert isinstance(p_repeat, torch.Tensor)
    assert len(p_repeat) == 2
    assert p_repeat[0] == -2.0
    assert p_repeat[1] == -1.0