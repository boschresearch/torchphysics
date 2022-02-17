import pytest
import torch
import matplotlib.pyplot as pyplot

import torchphysics.utils.plotting.plot_functions as plt  
from torchphysics.problem.domains import Interval, Circle, Parallelogram
from torchphysics.problem.spaces import R2, R1
from torchphysics.problem.samplers import PlotSampler
from torchphysics.models.fcn import FCN
from torchphysics.problem.spaces.points import Points


def plt_func(u):
    return torch.linalg.norm(u, dim=1)


def test_plot_create_info_text():
    input_p = Points(torch.tensor([[3.0, 34]]), R1('t')*R1('D'))
    text = plt._create_info_text(input_p)
    assert text == 't = 3.0\nD = 34.0'
    input_p = Points.empty()
    text = plt._create_info_text(input_p)
    assert text == ''


def test_plot_triangulation_of_domain():
    domain = Parallelogram(R2('x'), [0, 0], [1, 0.0], [0, 1])
    ps = PlotSampler(domain, n_points=200)
    domain_points = ps.sample_points()
    numpy_points = domain_points.as_tensor.detach().numpy()
    triangulation = plt._triangulation_of_domain(domain, numpy_points) 
    assert len(triangulation.x) == len(numpy_points)
    assert len(triangulation.y) == len(numpy_points)
    points = torch.column_stack((torch.FloatTensor(triangulation.x),
                                torch.FloatTensor(triangulation.y)))
    points = Points(points, R2('x'))
    assert all(torch.logical_or(domain._contains(points),
                                domain.boundary._contains(points)))


def test_Plotter():
    domain = Parallelogram(R2('x'), [0, 0], [1, 0.0], [0, 1])
    ps = PlotSampler(domain, n_points=200)
    plotter = plt.Plotter(plot_function=plt_func, point_sampler=ps)
    assert plotter.plot_function.fun == plt_func
    assert plotter.point_sampler == ps
    assert plotter.angle == [30, 30]
    assert plotter.log_interval == None
    assert plotter.plot_type == ''


def test_1D_plot():
    domain = Interval(R1('x'), 0, 1)
    ps = PlotSampler(domain, n_points=200)
    plotter = plt.Plotter(plot_function=plt_func, point_sampler=ps)
    model = FCN(input_space=R1('x'), output_space=R1('u'))
    fig = plotter.plot(model=model)  
    assert torch.allclose(torch.tensor(fig.axes[0].get_xlim()).float(),
                          torch.tensor((-0.05, 1.05)))
    assert fig.axes[0].get_xlabel() == 'x'
    pyplot.close(fig)


def test_line_plot_with_wrong_function_shape():
    def wrong_shape(u):
        return u
    domain = Interval(R1('x'), 0, 1)
    ps = PlotSampler(domain, n_points=200)
    plotter = plt.Plotter(plot_function=wrong_shape, point_sampler=ps, 
                          plot_type='line')
    model = FCN(input_space=R1('x'), output_space=R2('u'))
    with pytest.raises(ValueError):
        _ = plotter.plot(model=model)  


def test_1D_plot_with_textbox():
    domain = Interval(R1('x'), 0, 1)
    ps = PlotSampler(domain, n_points=200, data_for_other_variables={'t': 2})
    plotter = plt.Plotter(plot_function=plt_func, point_sampler=ps)
    model = FCN(input_space=R1('x')*R1('t'), output_space=R1('u'))
    fig = plotter.plot(model=model)  
    assert torch.allclose(torch.tensor(fig.axes[0].get_xlim()).float(),
                          torch.tensor((-0.05, 1.05)))
    assert fig.axes[0].get_xlabel() == 'x'
    pyplot.close(fig)


def test_2D_plot():
    domain = Parallelogram(R1('x')*R1('y'), [0, 0], [1, 0], [0, 2])
    ps = PlotSampler(domain, n_points=200)
    plotter = plt.Plotter(plot_function=plt_func, point_sampler=ps)
    model = FCN(input_space=R1('x')*R1('y'), output_space=R1('u'))
    fig = plotter.plot(model=model)  
    assert torch.allclose(torch.tensor(fig.axes[0].get_xlim()).float(),
                          torch.tensor((-0.05, 1.05)))
    assert fig.axes[0].get_xlabel() == 'x'
    assert torch.allclose(torch.tensor(fig.axes[0].get_ylim()).float(),
                          torch.tensor((-0.1, 2.1)))
    assert fig.axes[0].get_ylabel() == 'y'
    pyplot.close(fig)


def test_2D_plot_for_booleandomain():
    domain = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 2])
    domain -= Circle(R2('x'), [0.5, 0.5], 0.1)
    ps = PlotSampler(domain, density=10)
    plotter = plt.Plotter(plot_function=plt_func, point_sampler=ps)
    model = FCN(input_space=R2('x'), output_space=R1('u'))
    fig = plotter.plot(model=model)  
    assert torch.allclose(torch.tensor(fig.axes[0].get_xlim()).float(),
                          torch.tensor((-0.05, 1.05)))
    assert fig.axes[0].get_xlabel() == 'x_1'
    assert torch.allclose(torch.tensor(fig.axes[0].get_ylim()).float(),
                          torch.tensor((-0.1, 2.1)))
    assert fig.axes[0].get_ylabel() == 'x_2'
    pyplot.close(fig)


def test_scatter():
    #1D
    I = Interval(R1('x'), 0, 1)
    data = I.sample_grid(20)
    fig = plt._scatter(I.space, data)
    assert fig.axes[0].get_xlabel() == 'x'
    #2D
    I = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 2])
    data = I.sample_grid(20)
    fig = plt._scatter(I.space, data)
    assert fig.axes[0].get_xlabel() == 'x'
    assert fig.axes[0].get_ylabel() == 'x'
    #mixed
    R = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 2])
    I = Interval(R1('t'), 0, 1)
    domain = R * I
    data = domain.sample_random_uniform(20)
    fig = plt._scatter(domain.space, data)
    assert fig.axes[0].get_xlabel() == 'x'
    assert fig.axes[0].get_ylabel() == 'x'
    assert fig.axes[0].get_zlabel() == 't'
    pyplot.close(fig)


def test_2D_quiver():
    def quiver_plt(u):
        return u.detach().cpu().numpy()
    P = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 2])
    ps = PlotSampler(P, density=50)
    plotter = plt.Plotter(plot_function=quiver_plt, point_sampler=ps)
    model = FCN(input_space=R2('x'), output_space=R2('u'))
    fig = plotter.plot(model=model)  
    assert torch.allclose(torch.tensor(fig.axes[0].get_xlim()).float(),
                          torch.tensor((-0.05, 1.05)))
    assert fig.axes[0].get_xlabel() == 'x_1'
    assert torch.allclose(torch.tensor(fig.axes[0].get_ylim()).float(),
                          torch.tensor((-0.1, 2.1)))
    assert fig.axes[0].get_ylabel() == 'x_2'
    pyplot.close(fig)


def test_2D_quiver_with_textbox():
    def quiver_plt(u):
        return u.detach().cpu().numpy()
    P = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 2])
    ps = PlotSampler(P, density=50, data_for_other_variables={'t': 2.023223})
    plotter = plt.Plotter(plot_function=quiver_plt, point_sampler=ps)
    model = FCN(input_space=R2('x')*R1('t'), output_space=R2('u'))
    fig = plotter.plot(model=model)  
    assert torch.allclose(torch.tensor(fig.axes[0].get_xlim()).float(),
                          torch.tensor((-0.05, 1.05)))
    assert fig.axes[0].get_xlabel() == 'x_1'
    assert torch.allclose(torch.tensor(fig.axes[0].get_ylim()).float(),
                          torch.tensor((-0.1, 2.1)))
    assert fig.axes[0].get_ylabel() == 'x_2'
    pyplot.close(fig)


def test_3D_curve():
    I = Interval(R1('i'), -1, 2)
    ps = PlotSampler(I, density=50, data_for_other_variables={'t': 2})
    plotter = plt.Plotter(plot_function=lambda u:u, point_sampler=ps)
    model = FCN(input_space=R1('i')*R1('t'), output_space=R2('u'))  
    fig = plotter.plot(model=model)  
    assert torch.allclose(torch.tensor(fig.axes[0].get_xlim()).float(),
                          torch.tensor((-1.15, 2.15)))
    assert fig.axes[0].get_xlabel() == 'i'
    pyplot.close(fig)


def test_contour_2D():
    P = Parallelogram(R2('R'), [0, 0], [1, 0], [0, 2])
    model = FCN(input_space=R2('R'), output_space=R2('u'))    
    ps = PlotSampler(P, n_points=500)
    plotter = plt.Plotter(plot_function=plt_func, point_sampler=ps,
                          plot_type='contour_surface')
    fig = plotter.plot(model=model)  
    assert torch.allclose(torch.tensor(fig.axes[0].get_xlim()).float(),
                          torch.tensor((-0.05, 1.05)))
    assert fig.axes[0].get_xlabel() == 'R_1'
    assert torch.allclose(torch.tensor(fig.axes[0].get_ylim()).float(),
                          torch.tensor((-0.1, 2.1)))
    assert fig.axes[0].get_ylabel() == 'R_2'
    pyplot.close(fig)


def test_contour_2D_with_textbox():
    P = Parallelogram(R2('R'), [0, 0], [1, 0], [0, 2])
    model = FCN(input_space=R2('R')*R2('t'), output_space=R2('u'))    
    ps = PlotSampler(P, n_points=500, data_for_other_variables={'t': [2.0, 0.0]})
    fig  = plt.plot(model, plt_func, ps, plot_type='contour_surface')
    assert torch.allclose(torch.tensor(fig.axes[0].get_xlim()).float(),
                          torch.tensor((-0.05, 1.05)))
    assert fig.axes[0].get_xlabel() == 'R_1'
    assert torch.allclose(torch.tensor(fig.axes[0].get_ylim()).float(),
                          torch.tensor((-0.1, 2.1)))
    assert fig.axes[0].get_ylabel() == 'R_2'
    pyplot.close(fig)


def test_contour_plot_with_wrong_function_shape():
    def wrong_shape(u):
        return u
    P = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 2])
    ps = PlotSampler(P, n_points=200)
    plotter = plt.Plotter(plot_function=wrong_shape, point_sampler=ps, 
                          plot_type='contour_surface')
    model = FCN(input_space=R2('x'), output_space=R2('u'))
    with pytest.raises(ValueError):
        _ = plotter.plot(model=model)  