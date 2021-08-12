'''This file contains different functions for animating outputs of 
neural networks
'''
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib import animation as anim
import numpy as np
import torch
from . import plot 
from ..problem.domain.domain1D import Interval


def animation(model, solution_name, plot_variable, domain_points,
              animation_variable, frame_number, device='cpu', 
              ani_speed=50, angle=[30, 30], dic_for_other_variables=None,
              all_variables=None, plot_output_entries=[-1], 
              ani_type=''):
    '''Main function for animations
    
    Parameters
    ----------
    model : DiffEqModel
        A neural network of which the output should be animated.
    solution_name : str
        The output function of the network that should be animated.
    plot_variable : Variabale.
        The main variable over which the solution should be animated. 
    domain_points : int 
        The number of points that should be used for the domain discretization.
    animation_variable : Variable
        The variable over which the animation has to be created. Needs to 
        have an Interval as a domain.
    frame_number : int
        Number of frames, of the animation. Will be sampled equidistant in the 
        domain of the animation_variable.
    device : str or torch device
        The device of the model.
    ani_speed : Number, optional
        Speed of the animation.
    angle : list, optional
        The view angle for 3D plots. Standart angle is [30, 30]
    dic_for_other_variables : dict, optional
        A dictionary containing values for all the other variables of the 
        model. E.g. {'D' : [1,2], ...}
    all_variables : order dict or list, optional
        This dictionary should contain all variables w.r.t. the input order
        of the model. This gets automatically created when initializing the 
        setting. E.g. all_variables = Setting.variables.
        The input can also be a list of the varible names in the right order. 
        If the input is None, it is assumed that the order of the input is:
        (plot_variables, animation_variable, dic_for_other_variables(item_1),
         dic_for_other_variables(item_2), ...)   
    plot_output_entries : int or list, optional
        Specifies what outputs of the model should be animated. 
        If for example the output of the model is 3 dimensional, the first two outputs
        are some kind of vector field and the last output is a temperatur.
        Then for example set plot_output_entries = 2 to animate the temperatur over 
        the animation_variable, or set plot_output_entries = [0, 1] to create a
        animation of the vector field. Inputs like [0, 2] are also possible.
        If no entry is given, the method will try to use all outputs of the model.
    ani_type : str, optional
        Specifies how the output sholud be animated. If no input is given, the method
        will try to use a fitting way, to show the data. Implemented types are:
            - 'line' for line animations, with a 1D-domain and output
            - 'surface_2D' for surface animation, with a 2D-domain
            - 'curve' for a curve in 3D, with a 1D-domain, 
            - 'quiver_2D' for quiver/vector field animation, with a 2D-domain
            - 'contour_surface' for contour/colormaps, with a 2D-domain

    Returns
    -------
    plt.figure
        The figure handle of the created plot   
    animation.FuncAnimation
        The function that handles the animation  
    '''    
    if not isinstance(animation_variable.domain, Interval):
        raise ValueError('Domain of animation variable has to be an interval')
    plot_output_entries = plot._decide_plot_entries(model, solution_name,
                                                    plot_output_entries)
    # check if a plot type is specified
    ani_types = {'line': animation_line, 'surface_2D': animation_surface2D,
                 'curve': animation_curve_3D, 'quiver_2D': animation_quiver_2D, 
                 'contour_surface': animation_contour_2D}
    ani_fun = ani_types.get(ani_type)
    # check ourself if we can animated the input and output dimension
    if ani_fun is None:
    # If only one output should be used we create a line/surface animation
        if len(plot_output_entries) == 1:
            ani_fun = _animation_for_one_output(plot_variable)
        # If two outputs should be used we create a curve/quiver animation
        if len(plot_output_entries) == 2:
            ani_fun = _animation_for_two_outputs(plot_variable)
    if ani_fun is not None:
        return ani_fun(model=model, solution_name=solution_name,
                       plot_variable=plot_variable, points=domain_points,
                       angle=angle, ani_speed=ani_speed,
                       dic_for_other_variables=dic_for_other_variables,
                       all_variables=all_variables, frame_number=frame_number,
                       device=device, plot_output_entry=plot_output_entries, 
                       animation_variable=animation_variable)
    else:
        raise NotImplementedError(f"""Animations for a  
                                      {model.solution_dims[solution_name]} 
                                      dimensional output are not implemented!'  
                                      Please specify the output to animate.""")


def _animation_for_one_output(plot_variable):
    '''Handles animations if only one output of the model should be used.
    It will create a line or surface animation.
    '''
    # 2D animation (surface plot)
    if plot_variable.domain.dim == 2:
        return animation_surface2D
    # 1D animation (line plot):    
    elif plot_variable.domain.dim == 1:
        return animation_line
    else:
        raise NotImplementedError


def animation_line(model, solution_name, plot_variable, points,
                   animation_variable, frame_number,
                   device, ani_speed, angle, dic_for_other_variables,
                   all_variables, plot_output_entry):
    '''Handels 1D animations, inputs are the same as animation().
    '''
    # create input dic. for the model
    points, domain_points, input_dic, animation_points = \
        _create_input_for_animation(plot_variable, points, animation_variable,
                                    frame_number, device, dic_for_other_variables,
                                    all_variables)
    # evaluate the model and get max and min values over all points
    outputs = _evaluate_model(model, solution_name, points, animation_points, 
                              animation_variable.name, input_dic, plot_output_entry)
    outputs = plot._take_norm_of_output(plot_output_entry, outputs, axis=-1)
    output_max, output_min = _get_max_min(outputs)
    # construct the figure handle and axis for the animation
    fig = plt.figure()
    ax = plt.axes(xlim=(np.amin(domain_points), np.amax(domain_points)),
                  ylim=(output_min, output_max))
    ax.set_xlabel(plot_variable.name)
    ax.grid()
    line, = ax.plot([], [], lw=2)
    text_box = ax.text(0.05,0.95, '', bbox={'facecolor': 'w', 'pad': 5}, 
                       transform=ax.transAxes, va='top', ha='left')   
    dic_for_other_variables = _check_dic_is_none(dic_for_other_variables)
    # create the animation
    def animate(frame_number, outputs, line):
        line.set_data(domain_points.flatten(), outputs[:, frame_number, 0])
        dic_for_other_variables[animation_variable.name] = \
            animation_points[frame_number][0] 
        # set new text
        info_string = plot._create_info_text(dic_for_other_variables)  
        text_box.set_text(info_string)
    
    ani = anim.FuncAnimation(fig, animate, frames=frame_number, 
                             fargs=(outputs, line), interval=ani_speed)
    return fig, ani


def animation_surface2D(model, solution_name, plot_variable, points,
                        animation_variable,
                        frame_number, device, ani_speed, angle,
                        dic_for_other_variables, all_variables, plot_output_entry):
    '''Handels 2D animations, inputs are the same as animation().
    '''
    points, domain_points, input_dic, animation_points = \
        _create_input_for_animation(plot_variable, points, animation_variable,
                                    frame_number, device, dic_for_other_variables,
                                    all_variables)
    # evaluate the model and get max and min values over all points
    outputs = _evaluate_model(model, solution_name, points, animation_points, 
                              animation_variable.name, input_dic, plot_output_entry)
    output_max, output_min = _get_max_min(outputs)
    # triangulate the domain
    triangulation =  plot._triangulation_of_domain(plot_variable, domain_points)
    # construct the figure handle and axis for the animation
    fig, ax = plot._create_figure_and_axis(angle)
    plot._set_x_y_axis_data(plot_variable, ax)
    ax.set_zlim((output_min, output_max))
    text_box = ax.text2D(1.1, 0, '', bbox={'facecolor':'w', 'pad':5}, 
                         transform=ax.transAxes, va='top', ha='left')   
        
    # construct an auxiliary plot to get a fixed colorbar for the animation     
    surf = [ax.plot_surface(np.zeros((2, 2)),np.zeros((2, 2)),np.zeros((2, 2)), 
                            color='0.75', cmap=cm.jet, vmin=output_min,
                            vmax=output_max, antialiased=False)]
    plt.colorbar(surf[0], shrink=0.5, aspect=10, pad=0.1) 

    dic_for_other_variables = _check_dic_is_none(dic_for_other_variables)
    # create the animation
    def animate(frame_number, outputs, surf):
        surf[0].remove() # remove old surface
        surf[0] = ax.plot_trisurf(triangulation, outputs[:, frame_number, 0],
                                  color='0.75', cmap=cm.jet, 
                                  vmin=output_min, vmax=output_max, antialiased=False)
        # get new point auf the animation variable:
        new_animation_point = animation_points[frame_number][0] 
        dic_for_other_variables[animation_variable.name] = new_animation_point
        # set new text
        info_string = plot._create_info_text(dic_for_other_variables)  
        text_box.set_text(info_string)

    ani = anim.FuncAnimation(fig, animate, frames=frame_number, 
                             fargs=(outputs, surf), interval=ani_speed)

    return fig, ani


def _animation_for_two_outputs(plot_variable):
    '''Handles animations if two outputs of the model should be used.
    It will create a curve or quiver animation.
    '''
    # animate curve in 3D
    if plot_variable is None:
        return animation_curve_3D
    # animate quiver plot   
    elif plot_variable.domain.dim == 2:
        return animation_quiver_2D
    else:
        raise NotImplementedError


def animation_curve_3D(model, solution_name, plot_variable, points,
                       animation_variable, frame_number,
                       device, ani_speed, angle, dic_for_other_variables,
                       all_variables, plot_output_entry):
    '''Handles a curve animation, inputs are tha same as in animation
    '''
    # create the input dic for the model
    animation_points, input_dic = plot._create_domain(animation_variable,
                                                      frame_number, device)
    input_dic = plot._create_input_dic(input_dic, frame_number, dic_for_other_variables, 
                                       all_variables, device)
    
    # evaluate the model
    outputs = model.forward(input_dic)[solution_name].data.cpu().numpy()[:, plot_output_entry]

    # construct the figure handle and axis for the animation
    fig, ax = plot._create_figure_and_axis(angle)
    animation_points = animation_points.flatten()
    ax.set_xlim((animation_points[0], animation_points[-1]))
    ax.set_ylim((np.min(outputs[:, 0]), np.max(outputs[:, 0])))
    ax.set_zlim((np.min(outputs[:, 1]), np.max(outputs[:, 1])))
    ax.set_xlabel(animation_variable.name)
    line, = ax.plot([], [], [], lw=1)
    # again helper 'axis' to better show the curve:
    ax.plot(animation_points, np.zeros_like(animation_points),
            np.zeros_like(animation_points), color='black')

    dic_for_other_variables = _check_dic_is_none(dic_for_other_variables)
    # create the animation
    info_string = plot._create_info_text(dic_for_other_variables)  
    ax.text2D(1.1, 0, info_string, bbox={'facecolor':'w', 'pad':5}, 
              transform=ax.transAxes, va='top', ha='left') 

    def animate(frame_number, outputs, line):
        line.set_data(animation_points[0:frame_number], outputs[0:frame_number, 0])
        line.set_3d_properties(outputs[0:frame_number, 1])
        line.set_marker("o")

    ani = anim.FuncAnimation(fig, animate, frames=frame_number, 
                             fargs=(outputs, line), interval=ani_speed)
    
    return fig, ani


def animation_quiver_2D(model, solution_name, plot_variable, points,
                        animation_variable, frame_number, device,
                        ani_speed, angle,
                        dic_for_other_variables, all_variables, plot_output_entry):
    '''Handles quiver animations in 2D
    '''
    # create the input dic for the model
    points, domain_points, input_dic, animation_points = \
        _create_input_for_animation(plot_variable, points, animation_variable,
                                    frame_number, device, dic_for_other_variables,
                                    all_variables)
    # evaluate the model and get max and min values over all points
    outputs = _evaluate_model(model, solution_name, points, animation_points, 
                              animation_variable.name, input_dic, plot_output_entry)
    # for the colors
    color = np.linalg.norm(outputs, axis=-1)
    _, j = np.unravel_index(color.argmax(), color.shape)
    #outputs /= max_norm
    norm = colors.Normalize()
    norm.autoscale(color)
    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.grid()
    plot._set_x_y_axis_data(plot_variable, ax)
    text_box = ax.text(1.25, 0.5, '', bbox={'facecolor': 'w', 'pad': 5},
                       transform=ax.transAxes)
    plot._outline_domain(plot_variable, ax)
    # helper quiver plot to get a fixed colorbar and a constant scaling
    quiver = ax.quiver(domain_points[:, 0], domain_points[:, 1], 
                       outputs[:, j, 0], outputs[:, j, 1],
                       color=cm.jet(norm(color[:, 0])),
                       scale=None, angles='xy',
                       units='xy', zorder=10)
    sm = cm.ScalarMappable(cmap=cm.jet, norm=norm)
    quiver._init() # to fix the arrow scale
    plt.colorbar(sm)
    dic_for_other_variables = _check_dic_is_none(dic_for_other_variables)
    # create the animation
    def animate(frame_number, outputs, quiver):
        # set new coords. of arrow head and color
        quiver.set_UVC(outputs[:, frame_number, 0], outputs[:, frame_number, 1])
        quiver.set_color(cm.jet(norm(color[:, frame_number])))
        # get new point auf the animation variable:
        new_animation_point = animation_points[frame_number][0] 
        dic_for_other_variables[animation_variable.name] = new_animation_point
        # set new text
        info_string = plot._create_info_text(dic_for_other_variables)  
        text_box.set_text(info_string)
    
    ani = anim.FuncAnimation(fig, animate, frames=frame_number, 
                             fargs=(outputs, quiver), interval=ani_speed)

    return fig, ani


def animation_contour_2D(model, solution_name, plot_variable, points,
                         animation_variable, frame_number, device,
                         ani_speed, angle, dic_for_other_variables,
                         all_variables, plot_output_entry):
    '''Handles colormap animations in 2D
    '''
    # create the input dic for the model
    points, domain_points, input_dic, animation_points = \
        _create_input_for_animation(plot_variable, points, animation_variable,
                                    frame_number, device, dic_for_other_variables,
                                    all_variables)
    # evaluate the model and get max and min values over all points
    outputs = _evaluate_model(model, solution_name, points, animation_points, 
                              animation_variable.name, input_dic, plot_output_entry)
    outputs = plot._take_norm_of_output(plot_output_entry, outputs, axis=-1)
    output_max, output_min = _get_max_min(outputs)
    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.grid()
    plot._set_x_y_axis_data(plot_variable, ax)
    ax.set_xlabel(plot_variable.name + '_1')
    ax.set_ylabel(plot_variable.name + '_2')
    text_box = ax.text(1.3, 0.5, '', bbox={'facecolor': 'w', 'pad': 5},
                       transform=ax.transAxes)
    plot._outline_domain(plot_variable, ax)

    triangulation = plot._triangulation_of_domain(plot_variable, domain_points)
    # helper plot for fixed colorbar
    con = [ax.scatter([0, 0], [0, 1], c=[output_min, output_max],
                      vmin=output_min, vmax=output_max, cmap=cm.jet)]
    plt.colorbar(con[0])
    con[0].remove()
    dic_for_other_variables = _check_dic_is_none(dic_for_other_variables)
    # create the animation
    def animate(frame_number, outputs, con):
        # remove old contour
        if frame_number >= 1:
            for tp in con[0].collections:
                tp.remove()
        # set new contour
        con[0] = ax.tricontourf(triangulation, outputs[:, frame_number].flatten(),
                                100, cmap=cm.jet, vmin=output_min, vmax=output_max)
        # get new point of the animation variable:
        new_animation_point = animation_points[frame_number][0] 
        dic_for_other_variables[animation_variable.name] = new_animation_point
        # set new text
        info_string = plot._create_info_text(dic_for_other_variables)  
        text_box.set_text(info_string)
    
    ani = anim.FuncAnimation(fig, animate, frames=frame_number, 
                             fargs=(outputs, con), interval=ani_speed)

    return fig, ani


def _create_input_for_animation(plot_variable, points, animation_variable,
                                frame_number, device, dic_for_other_variables,
                                all_variables):
    # create the input dic for the model
    domain_points, input_dic = plot._create_domain(plot_variable, points, device)
    animation_points = plot._create_domain(animation_variable, frame_number, device)[0]
    points = len(domain_points) # for 2D domains the number of points may change
    input_dic[animation_variable.name] = animation_points[0][0] * \
                                         torch.ones((points, 1), device=device)
    input_dic = plot._create_input_dic(input_dic, points, dic_for_other_variables, 
                                       all_variables, device)
                                       
    return points, domain_points, input_dic, animation_points


def _check_dic_is_none(dic_for_other_variables):
    if dic_for_other_variables is None:
        dic_for_other_variables = {}
    return dic_for_other_variables


def _get_max_min(points):
    '''Returns the max and min value over all points. Needed to get a fixed y-(or z)axis.
    '''
    return np.amax(points), np.amin(points)


def _evaluate_model(model, solution_name, points, animation_points, animation_name,
                    input_dic, plot_output_entry):
    '''Evaluates the model at all domain and animation points

    Parameters
    ----------
    model : diffeqmodel
        The model
    points : np.array
        Number of points in the domain
    animation_points : np.array
        The points for the animation (e.g the points in time)
    animation_name : str
        The name of the animation variable
    input_dic : dic
        The input of the model
    plot_output_entries : int or list, optional
        Specifies what outputs of the model should be used. (see animation-method) 
    '''
    outputs = np.zeros((points, len(animation_points), len(plot_output_entry)))
    for i in range(len(animation_points)):
        # only need to change the animation varibale
        input_dic[animation_name] = animation_points[i][0]*torch.ones((points, 1))
        out = model(input_dic)[solution_name]
        outputs[:,i,:] = out.data.cpu().numpy()[:, plot_output_entry]
    return outputs