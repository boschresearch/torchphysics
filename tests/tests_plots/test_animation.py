import pytest
import torch
import numpy as np
import os
import matplotlib.pyplot as pyplot

import torchphysics.utils.plotting.animation as ani
from torchphysics.problem.domains import Interval, Circle, Parallelogram
from torchphysics.problem.spaces import R2, R1, R3, Points
from torchphysics.problem.samplers import AnimationSampler
from torchphysics.models.fcn import FCN
from torchphysics.utils.user_fun import UserFunction


def ani_func(u):
    return u


def test_create_animation_sampler():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], 2)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=20, n_points=20)
    assert ps.frame_number == 20
    assert ps.n_points == 20


def test_create_animation_sampler_error_with_wrong_animation_domain():
    C = Circle(R2('x'), [0, 0], 2)
    with pytest.raises(AssertionError):
        _ = AnimationSampler(plot_domain=C, animation_domain=C,
                             frame_number=20, n_points=20)


def test_animation_sampler_get_animation_key():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], 2)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=20, n_points=20)
    assert ps.animation_key == 't'


def test_animation_sampler_check_variable_dependencie():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], 2)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=20, n_points=20)
    assert ps.plot_domain_constant
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], lambda t: t+1)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=20, n_points=20)
    assert not ps.plot_domain_constant


def test_animation_sampler_check_variable_dependencie():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], 2)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=20, n_points=20)
    assert ps.plot_domain_constant
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], lambda t: t+1)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=20, n_points=20)
    assert not ps.plot_domain_constant


def test_animation_sampler_create_animation_points():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], 2)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=20, n_points=20)
    ani_points = ps.sample_animation_points()
    assert isinstance(ani_points, Points)
    assert len(ani_points) == ps.frame_number


def test_animation_sampler_create_plot_domain_points_independent():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], 2)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=20, n_points=20)
    domain_points = ps.sample_plot_domain_points(None)
    assert isinstance(domain_points, Points)
    assert len(domain_points) == len(ps)
    assert domain_points._t.requires_grad


def test_animation_sampler_create_plot_domain_points_independent_with_density():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], 2)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=20, density=50)
    domain_points = ps.sample_plot_domain_points(None)
    assert isinstance(domain_points, Points)
    assert len(domain_points) == len(ps)
    assert domain_points._t.requires_grad


def test_animation_sampler_create_plot_domain_points_dependent():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], lambda t: t+1)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=20, n_points=20)
    ani_points = ps.sample_animation_points()
    domain_points = ps.sample_plot_domain_points(ani_points)
    assert isinstance(domain_points, list)
    assert len(domain_points) == ps.frame_number
    for i in range(ps.frame_number):
        assert domain_points[i]._t.requires_grad


def test_create_animation_data_independent():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], 2)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=20, n_points=20)
    model = FCN(input_space=R2('x')*R1('t'), output_space=R1('u')) 
    animation_points, domain_points, outputs, out_shape = \
        ani._create_animation_data(model, ani_func, ps)
    assert isinstance(animation_points, Points)
    assert isinstance(outputs, list)
    assert len(outputs) == ps.frame_number
    assert isinstance(domain_points, Points)
    assert len(domain_points) >= ps.n_points
    assert out_shape == 1


def test_create_animation_data_independent_with_additional_variable():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], 2)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=20, n_points=20,
                          data_for_other_variables={'D': [0.0, 2.2]})
    model = FCN(input_space=R2('x')*R1('t')*R2('D'), output_space=R1('u')) 
    animation_points, domain_points, outputs, out_shape = \
        ani._create_animation_data(model, ani_func, ps)
    assert isinstance(animation_points, Points)
    assert isinstance(outputs, list)
    assert len(outputs) == ps.frame_number
    assert isinstance(domain_points, Points)
    assert len(domain_points) >= ps.n_points
    assert out_shape == 1


def test_create_animation_data_dependent():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], lambda t: t+1)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=20, n_points=20,
                          data_for_other_variables={'D': [0.0, 2.2]})
    model = FCN(input_space=R2('x')*R1('t')*R2('D'), output_space=R1('u')) 
    animation_points, domain_points, outputs, out_shape = \
        ani._create_animation_data(model, ani_func, ps)
    assert isinstance(animation_points, Points)
    assert isinstance(outputs, list)
    assert len(outputs) == ps.frame_number
    assert isinstance(domain_points, list)
    assert len(domain_points[0]) == ps.n_points
    assert out_shape == 1


def test_create_animation_data_dependent_and_with_density():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], lambda t: t+1)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=20, density=50,
                          data_for_other_variables={'D': [0.0, 2.2]})
    model = FCN(input_space=R2('x')*R1('t')*R2('D'), output_space=R1('u')) 
    animation_points, domain_points, outputs, out_shape = \
        ani._create_animation_data(model, ani_func, ps)
    assert isinstance(animation_points, Points)
    assert isinstance(outputs, list)
    assert len(outputs) == ps.frame_number
    assert isinstance(domain_points, list)
    assert out_shape == 1


def test_evaluate_animation_function():
    model = FCN(input_space=R2('x'), output_space=R1('u'))  
    inp_dict = {'x': torch.tensor([[0.0, 0.2]])}
    output = ani._evaluate_animation_function(model, UserFunction(ani_func), inp_dict)
    assert isinstance(output, np.ndarray)


def test_evaluate_animation_function_if_no_tensor_is_used():
    model = FCN(input_space=R2('x'), output_space=R1('u'))    
    inp_dict = {'x': torch.tensor([[0.0, 0.2]])}
    def ani_func_2(u):
        return 0.0
    output = ani._evaluate_animation_function(model, ani_func_2, inp_dict)
    assert output == 0.0


def test_animate_with_wrong_output_shape():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], 2)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=20, density=0.3)
    model = FCN(input_space=R2('x')*R1('t'), output_space=R3('u')) 
    with pytest.raises(NotImplementedError):
        ani.animate(model, ani_func, ps)


def test_line_animation_if_domain_changes():
    I = Interval(R1('t'), 0, 1)
    I2 = Interval(R1('x'), 0, lambda t:t+1)
    ps = AnimationSampler(plot_domain=I2, animation_domain=I,
                          frame_number=10, n_points=10, 
                          data_for_other_variables={'D': [1, 1.0]})
    model = FCN(input_space=R1('x')*R1('t')*R2('D'), output_space=R1('u'))  
    fig, animation = ani.animate(model, ani_func, ps)
    animation.save('test.gif')
    os.remove('test.gif')
    pyplot.close(fig)


def test_2d_surface_animation():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], 3)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=5, n_points=10, 
                          data_for_other_variables={'D': 0.3})
    model = FCN(input_space=R2('x')*R1('t')*R1('D'), output_space=R1('u'))  
    fig, animation = ani.animate(model, ani_func, ps)
    animation.save('test.gif')
    os.remove('test.gif')
    pyplot.close(fig)


def test_2d_surface_animation_if_domain_changes():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], lambda t: t+1)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=5, n_points=50)
    model = FCN(input_space=R2('x')*R1('t'), output_space=R1('u'))  
    fig, animation = ani.animate(model, ani_func, ps)
    animation.save('test.gif')
    os.remove('test.gif')
    pyplot.close(fig)


def test_2d_surface_animation_for_domain_operations():
    I = Interval(R1('t'), 0, 1)
    C = Parallelogram(R2('x'), [-4, -4], [4, -4], [-4, 4]) - \
            Circle(R2('x'), [0, 0], lambda t: t+1)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=2, density=1)
    model = FCN(input_space=R2('x')*R1('t'), output_space=R1('u')) 
    fig, animation = ani.animate(model, ani_func, ps)
    animation.save('test.gif')
    os.remove('test.gif')
    pyplot.close(fig)


def test_2d_quiver_animation():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], 3)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=5, n_points=10, 
                          data_for_other_variables={'D': 0.3})
    model = FCN(input_space=R2('x')*R1('t')*R1('D'), output_space=R2('u')) 
    fig, animation = ani.animate(model, ani_func, ps)
    animation.save('test.gif')
    os.remove('test.gif')
    pyplot.close(fig)
    assert ps.plot_domain_constant


def test_2d_quiver_animation_cant_use_chanceing_domain():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], lambda t: t)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=5, n_points=10, 
                          data_for_other_variables={'D': 0.3})
    model = FCN(input_space=R2('x')*R1('t')*R1('D'), output_space=R2('u')) 
    with pytest.raises(NotImplementedError):
        _ = ani.animate(model, ani_func, ps)


def test_2d_contour_animation():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], 3)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=5, n_points=10, 
                          data_for_other_variables={'D': 0.3})
    model = FCN(input_space=R2('x')*R1('t')*R1('D'), output_space=R1('u')) 
    fig, animation = ani.animate(model, ani_func, ps, ani_type='contour_surface')
    animation.save('test.gif')
    os.remove('test.gif')
    pyplot.close(fig)
    assert ps.plot_domain_constant


def test_2d_contour_animation_if_domain_changes():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], lambda t: t+1)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=5, n_points=50)
    model = FCN(input_space=R2('x')*R1('t'), output_space=R1('u')) 
    assert not ps.plot_domain_constant            
    fig, animation = ani.animate(model, ani_func, ani_sampler=ps, ani_type='contour_surface')
    animation.save('test.gif')
    os.remove('test.gif')
    pyplot.close(fig)


def test_2d_contour_animation_for_domain_operations():
    I = Interval(R1('t'), 0, 1)
    C = Parallelogram(R1('x')*R1('y'), [-4, -4], [4, -4], [-4, 4]) - \
            Circle(R1('x')*R1('y'), [0, 0], lambda t: t+1)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=2, density=0.5)
    model =FCN(input_space=R1('x')*R1('y')*R1('t'), output_space=R1('u')) 
    fig, animation = ani.animate(model, ani_func, ps, ani_type='contour_surface')
    animation.save('test.gif')
    os.remove('test.gif')
    pyplot.close(fig)
    assert not ps.plot_domain_constant


def test_line_animation():
    I = Interval(R1('t'), 0, 1)
    I2 = Interval(R1('x'), 0, 1)
    ps = AnimationSampler(plot_domain=I2, animation_domain=I,
                          frame_number=5, n_points=10)
    model = FCN(input_space=R1('x')*R1('t'), output_space=R1('u')) 
    fig, animation = ani.animate(model, ani_func, ps)
    animation.save('test.gif')
    os.remove('test.gif')
    pyplot.close(fig)