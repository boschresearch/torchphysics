'''This file contains different functions for animating outputs of 
neural networks
'''
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation as anim
import numpy as np
import torch
from . import plot
from ..problem.domain.domain1D import Interval


def animation(model, plot_variables, domain_points,
              animation_variable, frame_number, ani_speed=50, angle=[30, 30],
              dic_for_other_variables=None, all_variables=None):
    '''Main function for animations

    Parameters
    ----------
    model : DiffEqModel
        A neural network of which the output should be plotted
    plot_variables : Variabale or list of Variables.
        The main variable(s) over which the solution should be plotted. 
    domain_points : int 
        The number of points that should be used for the plot.
    animation_variable : Variable
        The varaiable over which the animation has to be created. Needs to 
        have an Interval as a domain.
    frame_number : int
        Number of frames
    ani_speed : Number
        Speed of the animation
    angle : list, optional
        The view angle for surface plots. Standart angle is [30, 30]
    dic_for_other_variables : dict, optional
        A dictionary containing values for all the other variables of the 
        model. E.g. {'D' : [1,2], ...}
    all_variables : order dict or list, optional
        This dictionary should contain all variables w.r.t. the input order
        of the model. This gets automatically created when initializing the 
        setting. E.g. all_variables = Setting.variables.
        The input can also be a list of the varible names in the right order. 
        If the input is None, it is assumed that the order of the input is:
        (plot_variables, animation_variable. 
         dic_for_other_variables(item_1), dic_for_other_variables(item_2), ...)   

    Returns
    -------
    plt.figure
        The figure handle of the created plot   
    animation.FuncAnimation
        The function that handles the animation  
    '''
    if not isinstance(animation_variable.domain, Interval):
        raise ValueError('Domain of animation variable has to be an interval')
    if not isinstance(plot_variables, list):
        plot_variables = [plot_variables]
    if len(plot_variables) == 1 and plot_variables[0].domain.dim == 2:
        return _animation2D(model, plot_variables[0], domain_points,
                            animation_variable, frame_number, ani_speed,
                            angle, dic_for_other_variables, all_variables)
    elif len(plot_variables) == 1 and plot_variables[0].domain.dim == 1:
        return _animation1D(model, plot_variables[0], domain_points,
                            animation_variable, frame_number, ani_speed,
                            dic_for_other_variables, all_variables)
    else:
        raise NotImplementedError


def _animation1D(model, plot_variable, points, animation_variable, frame_number,
                 ani_speed, dic_for_other_variables, all_variables):
    domain_points, input_dic = plot._create_domain(plot_variable, points)
    animation_points = plot._create_domain(animation_variable, frame_number)[0]

    input_dic[animation_variable.name] = animation_points[0][0]*torch.ones((points, 1))
    input_dic = plot._create_input_dic(input_dic, points,
                                       dic_for_other_variables, all_variables)

    outputs = _evaluate_model(model, points, animation_points,
                              animation_variable.name, input_dic)
    output_max, output_min = _get_max_min(outputs)
    # construct the figure handle and axis for the animation
    fig = plt.figure()
    ax = plt.axes(xlim=(np.amin(domain_points), np.amax(domain_points)),
                  ylim=(output_min, output_max))
    ax.set_xlabel(plot_variable.name)
    ax.grid()
    line, = ax.plot([], [], lw=2)
    text_box = ax.text(0.05, 0.95, '', bbox={'facecolor': 'w', 'pad': 5},
                       transform=ax.transAxes, va='top', ha='left')
    # create the animation
    if dic_for_other_variables is None:
        dic_for_other_variables = {}

    def animate(frame_number, outputs, line):
        line.set_data(domain_points.flatten(), outputs[:, frame_number])
        dic_for_other_variables[animation_variable.name] = animation_points[frame_number][0]
        info_string = plot._create_info_text(dic_for_other_variables)
        text_box.set_text(info_string)

    ani = anim.FuncAnimation(fig, animate, frames=frame_number,
                             fargs=(outputs, line), interval=ani_speed)
    return fig, ani


def _animation2D(model, plot_variable, points, animation_variable, frame_number,
                 ani_speed, angle, dic_for_other_variables, all_variables):
    domain_points, input_dic = plot._create_domain(plot_variable, points)
    animation_points = plot._create_domain(animation_variable, frame_number)[0]
    points = len(domain_points)

    input_dic[animation_variable.name] = animation_points[0][0]*torch.ones((points, 1))
    input_dic = plot._create_input_dic(input_dic, points,
                                       dic_for_other_variables, all_variables)

    outputs = _evaluate_model(model, points, animation_points,
                              animation_variable.name, input_dic)
    output_max, output_min = _get_max_min(outputs)
    triangulation = plot._triangulation_of_domain(plot_variable, domain_points)
    # construct the figure handle and axis for the animation
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(angle[0], angle[1])
    ax.set_xlim((np.min(domain_points[:, 0]), np.max(domain_points[:, 0])))
    ax.set_ylim((np.min(domain_points[:, 1]), np.max(domain_points[:, 1])))
    ax.set_zlim((output_min, output_max))
    ax.set_xlabel(plot_variable.name + '1')
    ax.set_ylabel(plot_variable.name + '2')
    text_box = ax.text2D(1.1, 0, '', bbox={'facecolor': 'w', 'pad': 5},
                         transform=ax.transAxes, va='top', ha='left')

    # construct an auxiliary plot to get a fixed colorbar for the animation
    surf = [ax.plot_surface(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                            color='0.75', cmap=cm.jet, vmin=output_min,
                            vmax=output_max, antialiased=False)]
    plt.colorbar(surf[0], shrink=0.5, aspect=10, pad=0.1)

    # create the animation
    if dic_for_other_variables is None:
        dic_for_other_variables = {}

    def animate(frame_number, outputs, surf):
        surf[0].remove()
        surf[0] = ax.plot_trisurf(triangulation, outputs[:, frame_number],
                                  color='0.75', cmap=cm.jet,
                                  vmin=output_min, vmax=output_max, antialiased=False)
        new_animation_point = animation_points[frame_number][0]
        dic_for_other_variables[animation_variable.name] = new_animation_point
        info_string = plot._create_info_text(dic_for_other_variables)
        text_box.set_text(info_string)

    ani = anim.FuncAnimation(fig, animate, frames=frame_number,
                             fargs=(outputs, surf), interval=ani_speed)

    return fig, ani


def _get_max_min(points):
    return np.amax(points), np.amin(points)


def _evaluate_model(model, points, animation_points, animation_name, input_dic):
    outputs = np.zeros((points, len(animation_points)))
    for i in range(len(animation_points)):
        input_dic[animation_name] = animation_points[i][0]*torch.ones((points, 1))
        out = model.forward(input_dic)
        outputs[:, i] = out.data.cpu().numpy().flatten()
    return outputs
