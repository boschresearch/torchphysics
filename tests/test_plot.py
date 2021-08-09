import pytest
import torch
import os
import numpy as np
from matplotlib import animation as matplotani
import matplotlib.pyplot as pyplot

import torchphysics.utils.plot as plt
import torchphysics.utils.animation as ani  
from torchphysics.problem.variables import Variable
from torchphysics.problem.domain.domain_operations import Cut
from torchphysics.problem.domain.domain1D import Interval
from torchphysics.problem.domain.domain2D import (Rectangle, Polygon2D) 
from torchphysics.models.fcn import SimpleFCN


def test_order_dic():
    input = {'a': 1, 'c': 3, 'b': 2}
    order = {'a': 1, 'b': 1, 'c': 1}
    order2 = {'c': 1, 'b': 1, 'a': 1}
    output = plt._order_input_dic(input, order)
    output2 = plt._order_input_dic(input, order2)
    i = 1
    for name in output:
        assert output[name] == i
        i = i + 1
    i = 3
    for name in output2:
        assert output2[name] == i
        i = i - 1


def test_plot_text():
    input = {'t': 3, 'D': 34}
    text = plt._create_info_text(input)
    assert text == 't = 3\nD = 34'
    input = {}
    text = plt._create_info_text(input)
    assert text == ''


def test_domain_creation():
    input = Variable(name='x', domain=Rectangle([0, 0], [1, 0], [0, 1]))
    domain_points, dic =  plt._create_domain(input, points=20, device='cpu')
    assert len(domain_points) >= 20
    assert torch.equal(dic['x'], torch.tensor(domain_points))
    assert dic['x'].device.type == 'cpu'


def test_create_dic_for_other_variables():
    points = 200
    input = {'t': 2, 'D': [2, 3, 4]}
    output = plt._create_dic_for_other_variables(points, input, device='cpu')
    assert torch.equal(output['t'], 2*torch.ones((points, 1)))
    assert torch.equal(output['D'], 
                       torch.FloatTensor([2, 3, 4] * np.ones((points, 3))))


def test_create_dic_for_other_variables_throw_exception():
    points = 200
    input = {'t': 2, 'D': 'not okay'}
    with pytest.raises(ValueError):
        _ = plt._create_dic_for_other_variables(points, input, device='cpu')


def test_create_input_dic():
    input_dic = {'x': 3}
    points = 20 
    output = plt._create_input_dic(input_dic, points, 
                                   dic_for_other_variables=None,
                                   all_variables=None,
                                   device='cpu')
    assert output == input_dic
    output = plt._create_input_dic(input_dic, points, 
                                   dic_for_other_variables={'t': 3},
                                   all_variables={'x': 1, 't': 1},
                                   device='cpu')
    assert output['x'] == 3
    assert torch.equal(output['t'], 3*torch.ones((20, 1)))                         


def test_triangulation_on_rect():
    domain = Rectangle([0, 0], [1, 0], [0, 1])
    input = Variable(name='x', domain=domain)
    domain_points, _ =  plt._create_domain(input, points=20, device='cpu')
    triangulation = plt._triangulation_of_domain(input, domain_points) 
    assert len(triangulation.x) == len(domain_points)
    assert len(triangulation.y) == len(domain_points)
    points = np.append(triangulation.x, triangulation.y, axis=0)
    assert np.logical_or(domain.is_inside(points), domain.is_on_boundary(points)).all()


def test_triangulation_on_Polygon2D():
    domain = Polygon2D(corners=[[0, 0], [0.25, 0], [0.25, 0.5], [0.75, 0.5],
                                [0.75, 0], [1, 0], [1, 1], [0, 1]])
    input = Variable(name='x', domain=domain)
    domain_points, _ =  plt._create_domain(input, points=20, device='cpu')
    triangulation = plt._triangulation_of_domain(input, domain_points) 
    assert len(triangulation.x) == len(domain_points)
    assert len(triangulation.y) == len(domain_points)


def test_Plotter():
    x = Variable(name='x', domain=Rectangle([0, 0], [1, 0], [0, 1]))
    plotter = plt.Plotter(plot_variables=x, points=20,
                          solution_name='test',
                          dic_for_other_variables={'t': 3}, 
                          all_variables=None, 
                          log_interval=5)
    assert x == plotter.plot_variables
    assert plotter.points == 20
    assert plotter.log_interval == 5
    assert plotter.all_variables is None
    assert plotter.dic_for_other_variables == {'t': 3}


def test_1D_plot():
    I = Variable(name='x', domain=Interval(0, 1))
    plotter = plt.Plotter(plot_variables=I, points=20, solution_name='u',
                          dic_for_other_variables={'t': 2})
    model = SimpleFCN(variable_dims={'x':1, 't':1},
                      solution_dims={'u':1},
                      width=5, depth=1)
    fig = plotter.plot(model=model)  
    assert fig.axes[0].get_xlim() == (-0.05, 1.05)
    assert fig.axes[0].get_xlabel() == 'x'
    # test without other variables
    plotter = plt.Plotter(plot_variables=I, points=10, solution_name='u')
    model = SimpleFCN(variable_dims={'x':1},
                      solution_dims={'u':1},
                      width=5, depth=1)
    fig = plotter.plot(model=model)  
    pyplot.close(fig)


def test_2D_plot():
    R = Variable(name='x', domain=Rectangle([0, 0], [1, 0], [0, 2]))
    plotter = plt.Plotter(plot_variables=R, points=20, solution_name='u',
                          dic_for_other_variables={'t': 2})
    model = SimpleFCN(variable_dims={'x':2, 't':1},
                      solution_dims={'u':1},
                      width=5, depth=1)
    fig = plotter.plot(model=model)  
    assert fig.axes[0].get_xlim() == (-0.05, 1.05)
    assert fig.axes[0].get_xlabel() == 'x_1'
    assert fig.axes[0].get_ylim() == (-0.1, 2.1)
    assert fig.axes[0].get_ylabel() == 'x_2'
    # test without other variables
    plotter = plt.Plotter(plot_variables=R, points=10, solution_name='u'
    )
    model = SimpleFCN(variable_dims={'x':2},
                      solution_dims={'u':1},
                      width=5, depth=1)
    fig = plotter.plot(model=model)  
    pyplot.close(fig)


def test_plot_2D_two_intervals():
    I1 = Variable(name='x', domain=Interval(0, 1))
    I2 = Variable(name='t', domain=Interval(0, 2))
    plotter = plt.Plotter(plot_variables=[I1, I2], points=20, solution_name='u',
                          dic_for_other_variables={'D': 2})
    model = SimpleFCN(variable_dims={'x':1, 't':1, 'D':1},
                      solution_dims={'u':1},
                      width=5, depth=1)
    fig = plotter.plot(model=model)  
    assert fig.axes[0].get_xlim() == (-0.05, 1.05)
    assert fig.axes[0].get_xlabel() == 'x'
    assert fig.axes[0].get_ylim() == (-0.1, 2.1)
    assert fig.axes[0].get_ylabel() == 't'   
    # test without other variables
    plotter = plt.Plotter(plot_variables=[I1, I2], points=20, solution_name='u')
    model = SimpleFCN(variable_dims={'x':1, 't':1},
                      solution_dims={'u':1},
                      width=5, depth=1)
    fig = plotter.plot(model=model)  
    pyplot.close(fig)


def test_scatter():
    #1D
    all_variables = ['x']
    I = Variable(name='x', domain=Interval(0, 1))
    data = {'x': torch.tensor(I.domain.sample_inside(20))}
    fig = plt._scatter(all_variables, data)
    assert fig.axes[0].get_xlabel() == 'x'
    #2D
    all_variables = ['x']
    R = Variable(name='x', domain=Rectangle([0, 0], [0, 1], [2, 0]))
    data = {'x': torch.tensor(R.domain.sample_inside(20))}
    fig = plt._scatter(all_variables, data)
    assert fig.axes[0].get_xlabel() == 'x'
    assert fig.axes[0].get_ylabel() == 'x'
    #mixed
    all_variables = ['x', 't']
    R = Variable(name='x', domain=Rectangle([0, 0], [0, 1], [2, 0]))
    I = Variable(name='t', domain=Interval(0, 2))
    data = {'x': torch.tensor(R.domain.sample_inside(20)), 
            't': torch.tensor(I.domain.sample_inside(20))}
    fig = plt._scatter(all_variables, data)
    assert fig.axes[0].get_xlabel() == 'x'
    assert fig.axes[0].get_ylabel() == 'x'
    assert fig.axes[0].get_zlabel() == 't'
    pyplot.close(fig)


def test_2D_quiver():
    R = Variable(name='x', domain=Rectangle([0, 0], [1, 0], [0, 2]))
    plotter = plt.Plotter(plot_variables=R, points=20, solution_name='u',
                          dic_for_other_variables={'t': 2})
    model = SimpleFCN(variable_dims={'x':2, 't':1},
                      solution_dims={'u':2},
                      width=5, depth=1)
    fig = plotter.plot(model=model)  
    assert fig.axes[0].get_xlim() == (-0.05, 1.05)
    assert fig.axes[0].get_xlabel() == 'x_1'
    assert fig.axes[0].get_ylim() == (-0.1, 2.1)
    assert fig.axes[0].get_ylabel() == 'x_2'
    # test without other variables
    plotter = plt.Plotter(plot_variables=R, points=10, solution_name='u',
                          plot_output_entries=[0,1])
    model = SimpleFCN(variable_dims={'x':2},
                      solution_dims={'u':2},
                      width=5, depth=1)
    fig = plotter.plot(model=model) 
    pyplot.close(fig)


def test_quiver_2D_for_complex_domain():
    C = Cut(Rectangle([0, 0], [1, 0], [0, 2]), Rectangle([0,0], [0.5, 0], [0, 0.5]))
    R = Variable(name='R', domain=C)
    model = SimpleFCN(variable_dims={'R': 2},
                      solution_dims={'u': 2},
                      width=5, depth=1)     
    fig = plt.quiver2D(model, plot_variables=[R], angle=0, solution_name='u',
                       points=10, all_variables=None, device='cpu',
                       plot_output_entry=[0, 1], dic_for_other_variables={})
    assert fig.axes[0].get_xlim() == (-0.05, 1.05)
    assert fig.axes[0].get_xlabel() == 'R_1'
    assert fig.axes[0].get_ylim() == (-0.1, 2.1)
    assert fig.axes[0].get_ylabel() == 'R_2'
    pyplot.close(fig)


def test_3D_curve():
    I = Variable(name='i', domain=Interval(-1, 2))
    plotter = plt.Plotter(plot_variables=I, points=20, solution_name='u',
                          dic_for_other_variables={'t': 2})
    model = SimpleFCN(variable_dims={'i': 1, 't':1},
                      solution_dims={'u': 2},
                      width=5, depth=1)  
    fig = plotter.plot(model=model)  
    assert fig.axes[0].get_xlim() == (-1.15, 2.15)
    assert fig.axes[0].get_xlabel() == 'i'
    # test without other variables
    plotter = plt.Plotter(plot_variables=I, points=10, solution_name='u',
                          plot_output_entries=-1)
    model = SimpleFCN(variable_dims={'i': 1},
                      solution_dims={'u': 2},
                      width=5, depth=1)
    fig = plotter.plot(model=model) 
    pyplot.close(fig)


def test_contour_2D():
    R = Variable(name='R', domain=Rectangle([0, 0], [1, 0], [0, 2]))
    model = SimpleFCN(variable_dims={'R': 2},
                      solution_dims={'u': 2},
                      width=5, depth=1)    
    fig = plt.contour_2D(model, plot_variables=[R], angle=0, solution_name='u',
                         points=10, all_variables=None,
                         device='cpu', plot_output_entry=[0], 
                         dic_for_other_variables=None)
    assert fig.axes[0].get_xlim() == (-0.05, 1.05)
    assert fig.axes[0].get_xlabel() == 'R_1'
    assert fig.axes[0].get_ylim() == (-0.1, 2.1)
    assert fig.axes[0].get_ylabel() == 'R_2'
    # with extra input 
    model = SimpleFCN(variable_dims={'R': 2, 'D': 2},
                      solution_dims={'u': 2},
                      width=5, depth=1)        
    fig = plt.contour_2D(model, plot_variables=[R], angle=0, solution_name='u',
                         points=10, all_variables=None, device='cpu',
                         plot_output_entry=[0,1], dic_for_other_variables={'D':[0,1]})
    pyplot.close(fig)


def test_contour_2D_for_complex_domain():
    C = Cut(Rectangle([0, 0], [1, 0], [0, 2]), Rectangle([0,0], [0.5, 0], [0, 0.5]))
    R = Variable(name='R', domain=C)
    model = SimpleFCN(variable_dims={'R': 2, 'D': 2},
                      solution_dims={'u': 2},
                      width=5, depth=1)        
    fig = plt._plot(model, plot_variables=[R], plot_type='contour_surface',
                    solution_name='u',
                    points=10, plot_output_entries=[0,1],
                    dic_for_other_variables={'D':[0,1]})
    assert fig.axes[0].get_xlim() == (-0.05, 1.05)
    assert fig.axes[0].get_xlabel() == 'R_1'
    assert fig.axes[0].get_ylim() == (-0.1, 2.1)
    assert fig.axes[0].get_ylabel() == 'R_2'
    pyplot.close(fig)


# Test animation

def test_get_min_max_ani():
    points = [[2, 3, 4, 5], [23, 1, 3, 90], [2, 12, 2, -4]]
    maxi, mini = ani._get_max_min(points)
    assert maxi == 90
    assert mini == -4


def test_errors_by_ani():
    I1 = Variable(name='x', domain=Interval(0, 1))
    I2 = Variable(name='t', domain=Interval(0, 2))
    R = Variable(name='R', domain=Rectangle([0, 0], [1, 0], [0, 1]))
    model = SimpleFCN(variable_dims={'x': 1, 't': 1},
                      solution_dims={'u': 4},
                      width=5, depth=1)    
    with pytest.raises(ValueError):
        _, _ = ani.animation(model, plot_variable=I1, domain_points=10,
                             solution_name='u',
                             animation_variable=R, frame_number = 1)  
    with pytest.raises(NotImplementedError):
        _, _ = ani.animation(model, plot_variable=I1, domain_points=10,
                             solution_name='u',
                             animation_variable=I2, frame_number = 1) 


def test_ani_1D():
    I1 = Variable(name='x', domain=Interval(0, 1))
    I2 = Variable(name='t', domain=Interval(0, 2))
    model = SimpleFCN(variable_dims={'x': 1, 't': 1},
                      solution_dims={'u': 1},
                      width=5, depth=1)        
    fig, animation = ani.animation(model, plot_variable=I1, domain_points=10,
                                   animation_variable=I2, solution_name='u',
                                   frame_number=4)
    assert isinstance(animation, matplotani.FuncAnimation)
    # with extra input 
    model = SimpleFCN(variable_dims={'x': 1, 't': 1, 'D': 2},
                      solution_dims={'u': 1},
                      width=5, depth=1)      
    fig, animation = ani.animation(model, plot_variable=I1, domain_points=10,
                                   solution_name='u',
                                   animation_variable=I2, frame_number=1, 
                                   dic_for_other_variables={'D': [1, 2]})
    animation.save('test.gif')
    os.remove('test.gif')
    pyplot.close(fig)
                            

def test_ani_2D():
    R = Variable(name='R', domain=Rectangle([0, 0], [1, 0], [0, 1]))
    I2 = Variable(name='t', domain=Interval(0, 2))
    model = SimpleFCN(variable_dims={'R': 2, 't': 1},
                      solution_dims={'u': 1},
                      width=5, depth=1)   
    fig, animation = ani.animation(model, plot_variable=R, domain_points=10,
                                   animation_variable=I2, solution_name='u',
                                   frame_number=1, plot_output_entries=0)
    assert isinstance(animation, matplotani.FuncAnimation)
    assert fig.axes[0].get_xlim() == (-0.05, 1.05)
    assert fig.axes[0].get_xlabel() == 'R_1'
    assert fig.axes[0].get_ylim() == (-0.05, 1.05)
    assert fig.axes[0].get_ylabel() == 'R_2'
    animation.save('test.gif')
    os.remove('test.gif')
    pyplot.close(fig)


def test_ani_quiver_2D():
    R = Variable(name='R', domain=Rectangle([0, 0], [1, 0], [0, 2]))
    I2 = Variable(name='t', domain=Interval(0, 2))
    model = SimpleFCN(variable_dims={'R': 2, 't': 1},
                      solution_dims={'u': 2, 'k':2},
                      width=5, depth=1)      
    fig, animation = ani.animation(model, plot_variable=R, domain_points=10,
                                   animation_variable=I2, solution_name='k',
                                   frame_number=1, plot_output_entries=[0, 1])
    assert isinstance(animation, matplotani.FuncAnimation)
    assert fig.axes[0].get_xlim() == (-0.05, 1.05)
    assert fig.axes[0].get_xlabel() == 'R_1'
    assert fig.axes[0].get_ylim() == (-0.1, 2.1)
    assert fig.axes[0].get_ylabel() == 'R_2'
    animation.save('test.gif')
    os.remove('test.gif')
    pyplot.close(fig)


def test_ani_quiver_2D_for_complex_domain():
    C = Cut(Rectangle([0, 0], [1, 0], [0, 2]), Rectangle([0,0], [0.5, 0], [0, 0.5]))
    R = Variable(name='R', domain=C)
    I2 = Variable(name='t', domain=Interval(0, 2))
    model = SimpleFCN(variable_dims={'R': 2, 't': 1},
                      solution_dims={'u': 2, 'k':2},
                      width=5, depth=1)   
    fig, animation = ani.animation(model, plot_variable=R, domain_points=10,
                                   animation_variable=I2, solution_name='u',
                                   frame_number=1, plot_output_entries=[1, 0])
    assert isinstance(animation, matplotani.FuncAnimation)
    assert fig.axes[0].get_xlim() == (-0.05, 1.05)
    assert fig.axes[0].get_xlabel() == 'R_1'
    assert fig.axes[0].get_ylim() == (-0.1, 2.1)
    assert fig.axes[0].get_ylabel() == 'R_2'
    pyplot.close(fig)


def test_ani_contour_2D():
    R = Variable(name='R', domain=Rectangle([0, 0], [1, 0], [0, 2]))
    I2 = Variable(name='t', domain=Interval(0, 2))
    model = SimpleFCN(variable_dims={'R': 2, 't': 1},
                      solution_dims={'u': 2, 'k':2},
                      width=5, depth=1)       
    fig, animation = ani.animation(model, plot_variable=R,
                                   domain_points=10,
                                   solution_name='u',
                                   animation_variable=I2, ani_speed=5,
                                   frame_number=2, device='cpu',
                                   plot_output_entries=[0,1], 
                                   ani_type='contour_surface')
    assert isinstance(animation, matplotani.FuncAnimation)
    assert fig.axes[0].get_xlim() == (-0.05, 1.05)
    assert fig.axes[0].get_xlabel() == 'R_1'
    assert fig.axes[0].get_ylim() == (-0.1, 2.1)
    assert fig.axes[0].get_ylabel() == 'R_2'
    animation.save('test.gif')
    os.remove('test.gif')
    pyplot.close(fig)


def test_ani_contour_2D_for_complex_domain():
    C = Cut(Rectangle([0, 0], [1, 0], [0, 2]), Rectangle([0,0], [0.5, 0], [0, 0.5]))
    R = Variable(name='R', domain=C)
    I2 = Variable(name='t', domain=Interval(0, 2))
    model = SimpleFCN(variable_dims={'R': 2, 't': 1},
                      solution_dims={'u': 2, 'k':2},
                      width=5, depth=1)     
    fig, animation = ani.animation_contour_2D(model, plot_variable=R,
                                              points=10, all_variables=None,
                                              animation_variable=I2,
                                              solution_name='u',
                                              ani_speed=1, angle=0,
                                              frame_number=1, device='cpu',
                                              plot_output_entry=[0], 
                                              dic_for_other_variables=None)
    assert isinstance(animation, matplotani.FuncAnimation)
    assert fig.axes[0].get_xlim() == (-0.05, 1.05)
    assert fig.axes[0].get_xlabel() == 'R_1'
    assert fig.axes[0].get_ylim() == (-0.1, 2.1)
    assert fig.axes[0].get_ylabel() == 'R_2'
    pyplot.close(fig)


def test_ani_curve_3D():
    I = Variable(name='t', domain=Interval(0, 2))
    model = SimpleFCN(variable_dims={'t': 1},
                      solution_dims={'u': 2, 'k':2},
                      width=5, depth=1)    
    fig, animation = ani.animation(model, plot_variable=None, domain_points=0,
                                   solution_name='u',
                                   animation_variable=I, frame_number=2)
    assert isinstance(animation, matplotani.FuncAnimation)
    assert fig.axes[0].get_xlim() == (0.0, 2.0)
    assert fig.axes[0].get_xlabel() == 't'
    animation.save('test.gif')
    os.remove('test.gif')
    # with extra input 
    model = SimpleFCN(variable_dims={'t': 1, 'D': 2},
                      solution_dims={'u': 2, 'k':2},
                      width=5, depth=1)        
    fig, animation = ani.animation(model, plot_variable=None, domain_points=0,
                                   solution_name='k',
                                   animation_variable=I, frame_number=2, 
                                   dic_for_other_variables={'D': [1, 2]})
    pyplot.close(fig)