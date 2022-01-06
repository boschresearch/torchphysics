import pytest
import torch
import numpy as np

from torchphysics.problem.domains.domain2D.circle import Circle
from torchphysics.problem.domains.domain1D.interval import Interval
from torchphysics.problem.domains.domain0D.point import Point
from torchphysics.problem.domains.domainoperations.product import ProductDomain
from torchphysics.problem.spaces.space import R2, R1
from torchphysics.problem.spaces.points import Points


def test_create_product_domain_independent():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], 2)
    P = C * I
    assert P.domain_a == C
    assert P.domain_b == I
    assert len(P.necessary_variables) == 0
    assert P._is_constant
    assert P.dim == 3
    assert P.space == R2('x')*R1('t')


def test_create_product_domain_dependent():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], lambda t: t+1)
    P = C * I
    assert P.domain_a == C
    assert P.domain_b == I
    assert len(P.necessary_variables) == 0
    assert not P._is_constant


def test_create_product_domain_cricular_dependent():
    I = Interval(R1('t'), 0, lambda x: x[:, 0])
    C = Circle(R2('x'), [0, 0], lambda t: t+1)
    with pytest.raises(AssertionError):
        C * I


def test_create_product_domain_wrong_order():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], lambda t: t+1)
    with pytest.raises(AssertionError):
        I * C


def test_create_product_get_warning_for_space_overlap():
    I = Interval(R1('x'), 0, 1)
    C = Circle(R2('x'), [0, 0], 2)
    with pytest.warns(UserWarning):
        C * I


def test_call_product():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], 2)
    P = C * I
    P(y=4) 


def test_call_product_and_set_domain_b_to_point():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], lambda t: t+1)
    P = C * I
    new_P = P(t=4) 
    assert isinstance(new_P, ProductDomain)
    assert isinstance(new_P.domain_b, Point)
    assert isinstance(new_P.domain_a, Circle)


def test_call_product_and_set_domain_a_to_point():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], 3)
    P = C * I
    new_P = P(x=[0, 0]) 
    assert isinstance(new_P, ProductDomain)
    assert isinstance(new_P.domain_a, Point)
    assert isinstance(new_P.domain_b, Interval)


def test_get_boundary_of_product():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], 3)
    P = C * I
    bound = P.boundary
    assert isinstance(bound.domain_a, ProductDomain)
    assert isinstance(bound.domain_b, ProductDomain)


def test_bounding_box_independent():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], 2)
    P = C * I
    assert np.allclose(P.bounding_box(), [-2, 2, -2, 2, 0, 1])


def test_bounding_box_dependent_but_with_given_data():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], lambda t: t+1)
    P = C * I
    data = Points(torch.tensor([[0.0], [1.0]]), R1('t'))
    assert np.allclose(P.bounding_box(data), [-2, 2, -2, 2, 0, 1])


def test_bounding_box_when_set():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], lambda t: t+1)
    P = C * I
    P.set_bounding_box([-2, 2, -2, 2, 0, 1])
    assert np.allclose(P.bounding_box(), [-2, 2, -2, 2, 0, 1])


def test_bounding_box_dependent():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], lambda t: t+1)
    P = C * I
    with pytest.warns(UserWarning):
        bounds = P.bounding_box()
    assert bounds[0] >= -2
    assert bounds[1] <= 2
    assert bounds[2] >= -2
    assert bounds[3] <= 2
    assert bounds[4] == 0
    assert bounds[5] == 1    


def test_product_volume_if_constant():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], 2)
    P = C * I
    assert torch.isclose(P._get_volume(), torch.tensor(4*np.pi))


def test_product_volume_if_volume_was_set_before():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], lambda t: 10*t)
    P = C * I
    P.set_volume(23.0)
    assert torch.isclose(torch.tensor(P.volume()), torch.tensor(23.0))


def test_product_volume_dependet_only_on_b():
    torch.manual_seed(0)
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], lambda t: t)
    P = C * I
    with pytest.warns(UserWarning):
        volume = P.volume()
    assert torch.isclose(volume, torch.tensor(0.9455), atol=0.0002)


def test_product_volume_domain_b_is_dependent_on_variables():
    torch.manual_seed(0)
    I = Interval(R1('t'), 0, lambda D: D)
    C = Circle(R2('x'), [0, 0], lambda t: t)
    P = C * I
    data = Points(torch.tensor([[1.0], [2.0]]), R1('D'))
    with pytest.warns(UserWarning):
        volume = P.volume(data)
    assert torch.allclose(volume, torch.tensor([[1.1566], [4.3523]]), atol=0.0002)


def test_product_volume_domain_a_is_dependent_on_variables():
    torch.manual_seed(0)
    I = Interval(R1('t'), 0, 2)
    C = Interval(R1('x'), lambda D: -D, lambda t: t+1)
    P = C * I
    data = Points(torch.tensor([[1.0], [2.0]]), R1('D'))
    with pytest.warns(UserWarning):
        volume = P.volume(data)
    assert torch.allclose(volume, torch.tensor([[6.3447, 7.0938]]), atol=0.0002)


def test_product_contains_independent():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], 4)
    points = Points(torch.tensor([[1, 1, 0.8], [0.0, 2, 0.1], [-5, -5, 0], 
                                  [0, 0, -4], [-10, 2, 3]]), R2('x')*R1('t'))
    P = C * I
    inside = P._contains(points)
    assert all(inside[:2])
    assert not any(inside[2:])


def test_product_contains_dependent():
    I = Interval(R1('t'), 0, 2)
    C = Circle(R2('x'), [0, 0], lambda t: t+1)
    points = Points(torch.tensor([[0.1, 0.1, 0.8], [0.0, 1.5, 1], [-5, -5, 0], 
                                  [0.0, 1.5, 0.1], [0, 4, 3]]), R2('x')*R1('t'))
    P = C * I
    inside = P._contains(points)
    assert all(inside[:2])
    assert not any(inside[2:])


def test_product_grid_sampling_not_implemented():
    I = Interval(R1('t'), 0, 2)
    C = Circle(R2('x'), [0, 0], lambda t: t+1)
    P = C * I
    with pytest.raises(NotImplementedError):
        P.sample_grid(n=40)


def test_product_random_uniform_sampling_independent_with_n():
    I = Interval(R1('t'), 0, 2)
    C = Circle(R2('x'), [0, 0], 3)
    P = C * I
    points = P.sample_random_uniform(n=100)
    assert points.as_tensor.shape == (100, 3)
    assert torch.all(P._contains(points))


def test_product_random_uniform_sampling_with_d():
    I = Interval(R1('t'), 0, 2)
    C = Circle(R2('x'), [0, 0], 3)
    P = C * I
    points = P.sample_random_uniform(d=12)
    assert torch.all(P._contains(points))


def test_product_random_uniform_sampling_dependent_with_n():
    I = Interval(R1('t'), 0, 2)
    C = Circle(R2('x'), [0, 0], lambda t: t+1)
    P = C * I
    points = P.sample_random_uniform(n=100)
    assert points.as_tensor.shape == (100, 3)
    assert torch.all(P._contains(points))


def test_product_random_uniform_sampling_with_additional_params():
    I = Interval(R1('t'), 0, 2)
    C = Circle(R2('x'), [0, 0], lambda t: t+1)
    P = C * I
    params = Points(torch.tensor([[4.5]]), R1('D'))
    points = P.sample_random_uniform(n=100, params=params)
    assert points.as_tensor.shape == (100, 3)
    assert torch.all(P._contains(points))


def test_product_random_uniform_sampling_only_one_iteration():
    I = Interval(R1('t'), 0, 2)
    C = Circle(R2('x'), [0, 0], lambda t: t+1)
    P = C * I
    points = P.sample_random_uniform(n=1)
    assert points.as_tensor.shape == (1, 3)