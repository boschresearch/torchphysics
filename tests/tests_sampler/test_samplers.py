import pytest
import torch
from torchphysics.problem.samplers.sampler_base import StaticSampler

from torchphysics.problem.spaces import R2, R1
from torchphysics.problem.domains import (Circle, Interval, Point, Parallelogram)
from torchphysics.problem.samplers import *
from torchphysics.problem.spaces.points import Points
from torchphysics.utils.user_fun import UserFunction


def filter_func(x):
    return x[:, 0] <= 0


def test_sampler_creation():
    ps = PointSampler(n_points=40, density=13)
    assert ps.n_points == 40
    assert ps.density == 13
    assert ps.filter_fn == None


def test_sampler_creation_with_filter():
    ps = PointSampler(n_points=410, density=15, filter_fn=lambda t: 2*t)
    assert ps.n_points == 410
    assert ps.density == 15
    assert isinstance(ps.filter_fn, UserFunction)


def test_sampler_len_for_n():
    ps = PointSampler(n_points=21)
    assert len(ps) == 21


def test_sampler_set_length():
    ps = PointSampler()
    ps.set_length(34)
    assert len(ps) == 34


def test_sampler_len_for_density_not_definied():
    ps = PointSampler(density=14)
    with pytest.raises(ValueError):
        len(ps)


def test_sampler_apply_filter():
    ps = PointSampler(filter_fn=lambda x: x>=0)
    test_points = Points(torch.tensor([[1.0, 1.0, 0.0], [-10, 2.3, 2.3],
                                       [0.1, 0.0, 0.0]]), R1('x')*R2('t'))
    filtered_points = ps._apply_filter(test_points)
    assert len(filtered_points) == 2
    assert torch.all(filtered_points[:, ['x']].as_tensor == \
                     torch.tensor([[1.0], [0.1]]))
    assert torch.all(filtered_points[:, ['t']].as_tensor == \
                     torch.tensor([[1.0, 0.0], [0.0, 0.0]]))


def test_sampler_get_iterator():
    ps = PointSampler()
    my_iter = iter(ps)
    assert my_iter == ps
    with pytest.raises(NotImplementedError):
        next(my_iter)


def test_sampler_iteration_check():
    ps = PointSampler()
    ps._check_iteration_number(3, 0)


def test_sampler_iteration_check_warning():
    ps = PointSampler()
    with pytest.warns(UserWarning):
        ps._check_iteration_number(10, 10)


def test_sampler_iteration_check_error():
    ps = PointSampler()
    with pytest.raises(RuntimeError):
        ps._check_iteration_number(23, 0)


def test_sampler_product():
    ps_1 = PointSampler()
    ps_2 = PointSampler()
    product = ps_1 * ps_2
    assert product.sampler_a == ps_1
    assert product.sampler_b == ps_2


def test_sampler_product_length():
    ps_1 = PointSampler(n_points=30)
    ps_2 = PointSampler(n_points=10)
    product = ps_1 * ps_2
    assert len(product) == 300


def test_sampler_sum():
    ps_1 = PointSampler()
    ps_2 = PointSampler()
    sampler_sum = ps_1 + ps_2
    assert sampler_sum.sampler_a == ps_1
    assert sampler_sum.sampler_b == ps_2


def test_sampler_sum_length():
    ps_1 = PointSampler(n_points=30)
    ps_2 = PointSampler(n_points=10)
    sampler_sum = ps_1 + ps_2
    assert len(sampler_sum) == 40


def test_sampler_append():
    ps_1 = PointSampler()
    ps_2 = PointSampler()
    sampler_append = ps_1.append(ps_2)
    assert sampler_append.sampler_a == ps_1
    assert sampler_append.sampler_b == ps_2


def test_sampler_append_length():
    ps_1 = PointSampler(n_points=30)
    ps_2 = PointSampler(n_points=30)
    sampler_append = ps_1.append(ps_2)
    assert len(sampler_append) == 30


def test_random_sampler():
    C = Circle(R2('x'), [0, 0], 3)
    ps = RandomUniformSampler(C, 50)
    points = ps.sample_points()
    assert points.as_tensor.shape == (50, 2)
    assert all(C.__contains__(points))


def test_random_sampler_density():
    C = Circle(R2('x'), [0, 0], 3)
    ps = RandomUniformSampler(C, density=15)
    points = ps.sample_points()
    assert all(C.__contains__(points))


def test_random_sampler_product():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], lambda t : t+1)
    ps_I = RandomUniformSampler(I, n_points=10)
    ps_C = RandomUniformSampler(C, n_points=20)
    ps = ps_C * ps_I
    points = ps.sample_points()
    assert points.as_tensor.shape == (200, 3)
    assert all(C.__contains__(points))


def test_random_sampler_product_with_density():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], lambda t : t+1)
    ps_I = RandomUniformSampler(I, n_points=10)
    ps_C = RandomUniformSampler(C, density=18)
    ps = ps_C * ps_I
    points = ps.sample_points()
    assert all(C.__contains__(points))


def test_random_sampler_with_filter_and_density():
    C = Circle(R2('x'), [0, 0], 3)
    ps = RandomUniformSampler(C, density=14, filter_fn=filter_func)
    points = ps.sample_points()
    assert torch.all(filter_func(x=points.as_tensor))
    assert all(C.__contains__(points))


def test_random_sampler_with_filter_and_n_and_without_params():
    C = Circle(R2('x'), [0, 0], 3)
    ps = RandomUniformSampler(C, n_points=20, filter_fn=filter_func)
    points = ps.sample_points()
    assert points.as_tensor.shape == (20, 2)
    assert torch.all(filter_func(x=points.as_tensor))
    assert all(C.__contains__(points))


def test_random_sampler_with_filter_and_n_and_with_params():
    I = Interval(R1('t'), 0, 2)
    C = Circle(R2('x'), [0, 0], lambda t: t+1)
    pi = RandomUniformSampler(I, n_points=10)
    ps = RandomUniformSampler(C, n_points=50, filter_fn=filter_func)
    ps *= pi 
    points = ps.sample_points()
    assert points[:, ['x']].as_tensor.shape == (500, 2)
    assert points[:, ['t']].as_tensor.shape == (500, 1)
    assert torch.all(filter_func(x=points.as_tensor))
    assert all(C.__contains__(points))


def test_grid_sampler():
    P = Parallelogram(R2('x'), [0, 0], [2, 0], [0, 1])
    ps = GridSampler(P, 50)
    points = ps.sample_points()
    assert points.as_tensor.shape == (50, 2)
    assert all(P.__contains__(points))


def test_grid_sampler_density():
    P = Parallelogram(R2('x'), [0, 0], [2, 0], [0, 1])
    ps = GridSampler(P, density=15)
    points = ps.sample_points()
    assert all(P.__contains__(points))


def test_grid_sampler_product():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], lambda t : t+1)
    ps_I = GridSampler(I, n_points=10)
    ps_C = GridSampler(C, n_points=20)
    ps = ps_C * ps_I
    points = ps.sample_points()
    assert points[:, ['x']].as_tensor.shape == (200, 2)
    assert points[:, ['t']].as_tensor.shape == (200, 1)
    assert all(C.__contains__(points))


def test_grid_sampler_sum():
    P = Point(R2('x'), [0, 0])
    C = Circle(R2('x'), [0, 0], 3)
    ps_1 = GridSampler(P, n_points=10)
    ps_2 = GridSampler(C, n_points=20)
    ps = ps_1 + ps_2
    points = ps.sample_points()
    assert points.as_tensor.shape == (30, 2)
    assert all(C.__contains__(points))
    assert len(ps) == 30


def test_grid_sampler_with_filter_and_density():
    C = Circle(R2('x'), [0, 0], 3)
    ps = GridSampler(C.boundary, density=14, filter_fn=filter_func)
    points = ps.sample_points()
    assert torch.all(filter_func(x=points.as_tensor))
    assert all(C.boundary.__contains__(points))


def test_grid_sampler_with_filter_and_n_and_without_params():
    C = Circle(R2('x'), [0, 0], 3)
    ps = GridSampler(C, n_points=20, filter_fn=filter_func)
    points = ps.sample_points()
    assert points.as_tensor.shape == (20, 2)
    assert torch.all(filter_func(x=points.as_tensor))
    assert all(C.__contains__(points))


def test_grid_sampler_with_filter_and_n_and_with_params():
    I = Interval(R1('D'), 0, 2)
    C = Circle(R2('x'), [0, 0], lambda D: D+1)
    pi = GridSampler(I, n_points=10)
    ps = GridSampler(C, n_points=50, filter_fn=filter_func)
    ps *= pi 
    points = ps.sample_points()
    assert points[:, ['x']].as_tensor.shape == (500, 2)
    assert points[:, ['D']].as_tensor.shape == (500, 1)
    assert torch.all(filter_func(x=points.as_tensor))
    assert all(C.__contains__(points))


def test_grid_sampler_with_filter_and_n_and_all_points_valid():
    def redudant_filter(x):
        return x[:, 0] <= 10
    C = Circle(R2('x'), [0, 0], 3)
    ps = GridSampler(C, n_points=20, filter_fn=redudant_filter)
    points = ps.sample_points()
    assert points.as_tensor.shape == (20, 2)
    assert torch.all(redudant_filter(x=points.as_tensor))
    assert all(C.__contains__(points))


def test_grid_sampler_with_impossible_filter_and_n():
    def impossible_filter(x):
        return x[:, 0] <= -10
    C = Circle(R2('x'), [0, 0], 3)
    ps = GridSampler(C, n_points=20, filter_fn=impossible_filter)
    with pytest.raises(RuntimeError):
        _ = ps.sample_points()


def test_grid_sampler_resample_grid_warning():
    def redudant_filter(x):
        return x[:, 0] <= 10
    C = Circle(R2('x'), [0, 0], 3)
    ps = GridSampler(C, n_points=10, filter_fn=redudant_filter)
    with pytest.warns(UserWarning):
        points = ps._resample_grid(Points.empty(), Points.empty(),
                                   ps.domain.sample_grid, device='cpu')
    assert points.as_tensor.shape == (110, 2)
    assert torch.all(redudant_filter(x=points.as_tensor))
    assert all(C.__contains__(points))


def test_grid_sampler_append_no_random_points():
    a = torch.ones((10, 1))
    ps = GridSampler(None, n_points=10)
    assert torch.equal(ps._append_random_points(a, None, device='cpu'), a)


def test_spaced_grid_sampler():
    I = Interval(R1('t'), 0, 1)
    ps = ExponentialIntervalSampler(I, 50, exponent=2)
    points = ps.sample_points()
    assert points.as_tensor.shape == (50, 1)
    assert all(I.__contains__(points))


def test_spaced_grid_sampler_wrong_domain_type():
    P = Parallelogram(R2('x'), [0, 0], [2, 0], [0, 1])
    with pytest.raises(AssertionError):
        _ = ExponentialIntervalSampler(P, 50, exponent=2)


def test_spaced_grid_sampler_product():
    I = Interval(R1('t'), 0, 1)
    I_2 = Interval(R1('x'), 0, lambda t : t+1)
    ps_I = GridSampler(I, n_points=10)
    ps_I_2 = ExponentialIntervalSampler(I_2, n_points=20, exponent=3.2)
    ps = ps_I_2 * ps_I
    points = ps.sample_points()
    assert len(ps) == 200
    assert points.as_tensor.shape == (200, 2)
    assert all(I_2.__contains__(points))


def test_spaced_grid_sampler_append():
    I = Interval(R1('t'), 0, 1)
    I_2 = Interval(R1('x'), 0, 2)
    ps_I = GridSampler(I, n_points=10)
    ps_I_2 = ExponentialIntervalSampler(I_2, n_points=10, exponent=0.4)
    ps = ps_I.append(ps_I_2)
    points = ps.sample_points()
    assert points[:, ['x']].as_tensor.shape == (10, 1)
    assert points[:, ['t']].as_tensor.shape == (10, 1)
    assert all(I_2.__contains__(points))
    assert all(I.__contains__(points))
    assert len(ps) == 10


# plot samplers

def test_plot_sampler_creation():
    C = Circle(R2('x'), [0, 0], 3)
    ps = PlotSampler(C, n_points=34)
    assert len(ps) == 34
    assert ps.domain.center.fun == [0, 0]
    assert ps.domain.radius.fun == 3
    assert isinstance(ps.sampler, ConcatSampler)


def test_plot_sampler_creation_only_grid_samplers_used():
    C = Circle(R2('x'), [0, 0], 3)
    ps = PlotSampler(C, n_points=34)
    assert isinstance(ps.sampler.sampler_a, GridSampler)
    assert isinstance(ps.sampler.sampler_b, GridSampler)


def test_plot_sampler_creation_with_density():
    C = Circle(R2('x'), [0, 0], 3)
    ps = PlotSampler(C, density=14)
    assert ps.domain.center.fun == [0, 0]
    assert ps.domain.radius.fun == 3
    assert isinstance(ps.sampler, ConcatSampler)


def test_plot_sampler_creation_with_interval():
    I = Interval(R1('t'), 0, 2)
    ps = PlotSampler(I, n_points=34)
    assert isinstance(ps.sampler, ConcatSampler)
    assert isinstance(ps.sampler.sampler_a, ConcatSampler)
    assert isinstance(ps.sampler.sampler_b, GridSampler)


def test_plot_sampler_creation_with_intrval_with_density():
    I = Interval(R1('t'), 0, 2)
    ps = PlotSampler(I, density=13)
    assert isinstance(ps.sampler, ConcatSampler)
    assert isinstance(ps.sampler.sampler_a, ConcatSampler)
    assert isinstance(ps.sampler.sampler_b, GridSampler)


def test_plot_sampler_creation_for_variable_domain():
    C = Circle(R2('x'), [0, 0], lambda t: t+1)
    ps = PlotSampler(C, n_points=34, 
                     data_for_other_variables={'t': 1.0, 'D': [9.0, 2.0], 
                                               'r': torch.tensor([[0.2]])})
    assert len(ps) == 34
    assert ps.domain.center.fun == [0, 0]
    assert ps.domain.radius.fun == 2


def test_plot_sampler_creation_for_variable_domain_with_point_data():
    C = Circle(R2('x'), [0, 0], lambda t: t+1)
    data = Points(torch.tensor([[1.0, 9, 2.0]]), R1('t')*R2('D'))
    ps = PlotSampler(C, n_points=34, data_for_other_variables=data)
    assert len(ps) == 34
    assert ps.domain.center.fun == [0, 0]
    assert ps.domain.radius.fun == 2


def test_plot_sampler_create_points():
    C = Circle(R2('x'), [0, 0], 3)
    ps = PlotSampler(C, n_points=34)
    points = ps.sample_points()
    in_C = C._contains(points)
    on_C = C.boundary._contains(points)
    assert all(torch.logical_or(in_C, on_C))


## Test Gaussian sampler

def test_gaussian_sampler():
    I = Interval(R1('t'), 0, 1)
    ps = GaussianSampler(I, 50, mean=0.2, std=0.1)
    points = ps.sample_points()
    assert points.as_tensor.shape == (50, 1)
    assert all(I.__contains__(points))


def test_gaussian_sampler_in_2D():
    P = Parallelogram(R2('x'), [0, 0], [2, 0], [0, 1])
    ps = GaussianSampler(P, 100, mean=[0.2, 0.3], std=0.1)
    points = ps.sample_points()
    assert points.as_tensor.shape == (100, 2)
    assert all(P.__contains__(points))


def test_gaussian_sampler_wrong_domain_type():
    P = Parallelogram(R2('x'), [0, 0], [2, 0], [0, 1])
    with pytest.raises(AssertionError):
        _ = GaussianSampler(P.boundary, 50, mean=[0, 0], std=0.2)


def test_gaussian_sampler_wrong_mean_dimension():
    P = Parallelogram(R2('x'), [0, 0], [2, 0], [0, 1])
    with pytest.raises(AssertionError):
        _ = GaussianSampler(P, 50, mean=torch.tensor([0, 0, 0.3]), std=0.2)


def test_gaussian_sampler_product():
    I = Interval(R1('t'), 0, 1)
    I_2 = Interval(R1('x'), 0, lambda t : t+1)
    ps_I = GridSampler(I, n_points=10)
    ps_I_2 = GaussianSampler(I_2, n_points=20, mean=1, std=0.3)
    ps = ps_I_2 * ps_I
    points = ps.sample_points()
    assert points[:, ['x']].as_tensor.shape == (200, 1)
    assert points[:, ['t']].as_tensor.shape == (200, 1)
    assert all(I_2.__contains__(points))


def test_gaussian_sampler_product_in_2D():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], 2)
    ps_I = GridSampler(I, n_points=10)
    ps_I_2 = GaussianSampler(C, n_points=20, mean=[-2, 0], std=0.3)
    ps = ps_I_2 * ps_I
    points = ps.sample_points()
    assert points[:, ['x']].as_tensor.shape == (200, 2)
    assert points[:, ['t']].as_tensor.shape == (200, 1)
    assert all(C.__contains__(points))



## Test LHS sampler

def test_lhs_sampler():
    I = Interval(R1('t'), 0, 1)
    ps = LHSSampler(I, 50)
    points = ps.sample_points()
    assert points.as_tensor.shape == (50, 1)
    assert all(I.__contains__(points))


def test_lhs_sampler_in_2D():
    P = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    ps = LHSSampler(P, 100)
    points = ps.sample_points()
    assert points.as_tensor.shape == (100, 2)
    assert all(P.__contains__(points))


def test_lhs_sampler_wrong_domain_type():
    P = Parallelogram(R2('x'), [0, 0], [2, 0], [0, 1])
    with pytest.raises(AssertionError):
        _ = GaussianSampler(P.boundary, 50, mean=[0, 0], std=0.2)


def test_lhs_sampler_product():
    I = Interval(R1('t'), 0, 1)
    I_2 = Interval(R1('z'), 0, lambda t : t+1)
    ps_I = GridSampler(I, n_points=10)
    ps_I_2 = LHSSampler(I_2, n_points=20)
    ps = ps_I_2 * ps_I
    points = ps.sample_points()
    assert points[:, ['z']].as_tensor.shape == (200, 1)
    assert points[:, ['t']].as_tensor.shape == (200, 1)
    assert all(I_2.__contains__(points))


def test_lhs_sampler_product_in_2D():
    I = Circle(R1('t')*R1('y'), [0,0], 1)
    C = Circle(R2('x'), [0, 0], 2)
    ps_I = GridSampler(I, n_points=10)
    ps_I_2 = LHSSampler(C, n_points=20)
    ps = ps_I_2 * ps_I
    points = ps.sample_points()
    assert points.as_tensor.shape == (200, 4)
    assert all(C.__contains__(points))


# Test datasamplers:

def test_create_data_sampler_with_dict():#
    input_data = {'x': torch.tensor([[0.0, 0.0], [1.0, 2.0]])}
    ps = DataSampler(input_data)
    i = ps.sample_points()
    assert isinstance(i, Points)
    assert torch.equal(input_data['x'], i.as_tensor)


def test_create_data_sampler_with_points():
    input_data = Points(torch.tensor([[0.0, 0.0], [1.0, 2.0]]), R2('y'))
    ps = DataSampler(input_data)
    i  = ps.sample_points()
    assert isinstance(i, Points)
    assert i == input_data


def test_create_data_sampler_with_wrong_data_input():
    input_data = 'wrong_input'
    with pytest.raises(TypeError):
        DataSampler(input_data)


def test_create_data_sampler_with_wrong_data_output():
    output_data = 'wrong_output'
    input_data = Points(torch.tensor([[0.0], [2.0]]), R1('t'))
    with pytest.raises(TypeError):
        DataSampler(input_data, output_data)


# test static sampler

def test_make_sampler_static():
    ps = RandomUniformSampler(domain=Interval(R1('t'), 0, 2))
    assert not ps.is_static
    static_sampler = ps.make_static()
    assert isinstance(static_sampler, StaticSampler)
    assert static_sampler.is_static


def test_try_make_static_sampler_static_again():
    ps = RandomUniformSampler(domain=Interval(R1('t'), 0, 2))
    static_sampler = ps.make_static()
    assert static_sampler == static_sampler.make_static()


def test_get_points_of_static_sampler():
    ps = RandomUniformSampler(domain=Interval(R1('t'), 0, 2), n_points=50)
    static_sampler = ps.make_static()
    points1 = static_sampler.sample_points()
    points2 = static_sampler.sample_points()
    assert torch.equal(points1, points2)


def test_get_length_of_static_sampler():
    ps = RandomUniformSampler(domain=Interval(R1('t'), 0, 2), n_points=50)
    static_sampler = ps.make_static()
    assert len(static_sampler) == 50


def test_set_length_of_static_sampler():
    ps = RandomUniformSampler(domain=Interval(R1('t'), 0, 2), n_points=50)
    static_sampler = ps.make_static()
    static_sampler.set_length(100)
    assert len(static_sampler) == 100


def test_static_sampler_next():
    ps = RandomUniformSampler(domain=Interval(R1('t'), 0, 2), n_points=50)
    static_sampler = ps.make_static()
    points1 = next(static_sampler)
    points2 = next(static_sampler)
    assert torch.equal(points1, points2)