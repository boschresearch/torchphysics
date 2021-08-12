'''This file contains different functions for plotting outputs of
neural networks
'''
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.patches as patches
import scipy.spatial
import numpy as np
import torch
import numbers
from matplotlib import cm, colors


class Plotter():
    '''Object to collect plotting properties

    Parameters
    ----------
    solution_name : str
        The output function of the network that should be animated.
    plot_variables : Variabale or list of Variables.
        #TODO: what happens if dim(var) >= 3?
        The main variable(s) over which the solution should be plotted.
    points : int
        The number of points that should be used for the plot.
    dic_for_other_variables : dict, optional
        A dictionary containing values for all the other variables of the
        model. E.g. {'t' : 1, 'D' : [1,2], ...}
    all_variables : order dict or list, optional
        This dictionary should contain all variables w.r.t. the input order
        of the model. This gets automatically created when initializing the
        setting. E.g. all_variables = Setting.variables.
        The input can also be a list of the varible names in the right order.
        If the input is None, it is assumed that the order of the input is:
        (plot_variables, dic_for_other_variables(item_1),
         dic_for_other_variables(item_2), ...)
    log_interval : int
        Plots will be saved every log_interval steps if the plotter is used in
        training of a model.
    plot_output_entries : int or list, optional
        Specifies what outputs of the model should be plotted. 
        If for example the output of the model is 3 dimensional, the first two outputs
        are some kind of vector field and the last output is a temperatur.
        Then for example set plot_output_entries = 2 to plot the temperatur over 
        the other variables, or set plot_output_entries = [0, 1] to create a
        plot of the vector field. Inputs like [0, 2] are also possible.
        If no entry is given, the method will try to use all outputs of the model.
    plot_type : str, optional
        Specifies how the output sholud be plotted. If no input is given, the method
        will try to use a fitting way, to show the data. Implemented types are:
            - 'line' for plots in 1D
            - 'surface_2D' for surface plots, with a 2D-domain
            - 'curve' for a curve in 3D, with a 1D-domain, 
            - 'quiver_2D' for quiver/vector field plots, with a 2D-domain
            - 'contour_surface' for contour/colormaps, with a 2D-domain
    '''

    def __init__(self, solution_name, plot_variables, points, angle=[30, 30],
                 dic_for_other_variables=None, all_variables=None,
                 log_interval=None, plot_output_entries=[-1],
                 plot_type=''):
        self.solution_name = solution_name
        self.plot_variables = plot_variables
        self.points = points
        self.angle = angle
        self.dic_for_other_variables = dic_for_other_variables
        self.all_variables = all_variables
        self.log_interval = log_interval
        self.plot_output_entries = plot_output_entries
        self.plot_type = plot_type

    def plot(self, model, device='cpu'):
        return _plot(model, self.solution_name, self.plot_variables, self.points,
                     angle=self.angle,
                     dic_for_other_variables=self.dic_for_other_variables,
                     all_variables=self.all_variables,
                     device=device, plot_output_entries=self.plot_output_entries,
                     plot_type=self.plot_type)


def _plot(model, solution_name, plot_variables, points, angle=[30, 30],
          dic_for_other_variables=None, all_variables=None, device='cpu', 
          plot_output_entries=[-1], plot_type=''):
    '''Main function for plotting

    Parameters
    ----------
    model : DiffEqModel
        A neural network of which the output should be plotted
    solution_name : str
        The output function of the network that should be animated.
    plot_variables : Variabale or list of Variables.
        The main variable(s) over which the solution should be plotted.
    points : int
        The number of points that should be used for the plot.
    angle : list, optional
        The view angle for surface plots. Standart angle is [30, 30]
    dic_for_other_variables : dict, optional
        A dictionary containing values for all the other variables of the
        model. E.g. {'t' : 1, 'D' : [1,2], ...}
    all_variables : order dict or list, optional
        This dictionary should contain all variables w.r.t. the input order
        of the model. This gets automatically created when initializing the
        setting. E.g. all_variables = Setting.variables.
        The input can also be a list of the varible names in the right order.
        If the input is None, it is assumed that the order of the input is:
        (plot_variables, dic_for_other_variables(item_1),
         dic_for_other_variables(item_2), ...)
    device : str or torch device
        The device of the model.
    plot_output_entries : int or list, optional
        Specifies what outputs of the model should be plotted.
        If for example the output of the model is 3 dimensional, the first two outputs
        are some kind of vector field and the last output is a temperature.
        Then set for example plot_output_entries = 2 to plot the temperature over
        the other variables, or set plot_output_entries = [0, 1] to create a
        plot of the vector field. Inputs like [0, 2] are also possible.
        If no entry is given, the method will try to use all outputs of the model.
    plot_type : str, optional
        Specifies how the output sholud be plotted. If no input is given, the method
        will try to use a fitting way, to show the data. Implemented types are:
            - 'line' for plots in 1D
            - 'surface_2D' for surface plots, with a 2D-domain
            - 'curve' for a curve in 3D, with a 1D-domain, 
            - 'quiver_2D' for quiver/vector field plots, with a 2D-domain
            - 'contour_surface' for contour/colormaps, with a 2D-domain

    Returns
    -------
    plt.figure
        The figure handle of the created plot
    '''
    plot_output_entries = _decide_plot_entries(model, solution_name,
                                               plot_output_entries)
    # set everything to list to easier handle different cases
    if not isinstance(plot_variables, list):
        plot_variables = [plot_variables]
    # check if a plot type is specified
    plot_types = {'line': line_plot, 'surface_2D': surface2D,
                  'curve': curve3D, 'quiver_2D': quiver2D,
                  'contour_surface': contour_2D}
    plot_fun = plot_types.get(plot_type)
    # If no (or wrong) type is given, try to find the correct type:
    if plot_fun is None:
        if len(plot_output_entries) == 1:
            plot_fun = _plot_for_one_output(plot_variables)
        if len(plot_output_entries) == 2:
            plot_fun = _plot_for_two_outputs(plot_variables)
    if plot_fun is not None:
        return plot_fun(model=model, solution_name=solution_name,
                        plot_variables=plot_variables,
                        points=points, angle=angle,
                        dic_for_other_variables=dic_for_other_variables,
                        all_variables=all_variables,
                        device=device, plot_output_entry=plot_output_entries)
    else:
        raise NotImplementedError(f"""Plotting for a  
                                      {model.solution_dims[solution_name]} 
                                      dimensional output are not implemented!'  
                                      Please specify the output to plot.""")


def _plot_for_one_output(plot_variables):
    '''Handles plots if only one output of the model should be plotted.
    '''
    # surface plots:
    if len(plot_variables) == 1 and plot_variables[0].domain.dim == 2:
        return surface2D
    # line plots:                       
    elif len(plot_variables) == 1 and plot_variables[0].domain.dim == 1:
        return line_plot
    # surface plots for two different variables:
    elif (len(plot_variables) == 2 and
          plot_variables[0].domain.dim + plot_variables[1].domain.dim == 2):
        return surface2D_2_variables
    else:
        raise NotImplementedError("""Can't plot 1D-output on given domain""")


def surface2D(model, solution_name, plot_variables, points, angle,
              dic_for_other_variables, all_variables, device, plot_output_entry):
    '''Handels surface plots w.r.t. a two dimensional variable.
    Inputs are the same as _plot().
    '''
    # First create the input dic. for the model
    plot_variable = plot_variables[0]
    domain_points, input_dic = _create_input_points(points, dic_for_other_variables,
                                                    all_variables, device,
                                                    plot_variable)
    output = _evaluate_model(model, solution_name, plot_output_entry, input_dic)
    # For complex domains it is best to triangulate them for the plotting
    triangulation = _triangulation_of_domain(plot_variable, domain_points)
    fig, ax = _create_figure_and_axis(angle)
    _set_x_y_axis_data(plot_variable, ax)
    surf = ax.plot_trisurf(triangulation, output.flatten(),
                           cmap=cm.jet, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.4, aspect=5, pad=0.1)
    _add_textbox(dic_for_other_variables, ax, 1.2, 0.1)
    #plt.show()
    return fig


def line_plot(model, solution_name, plot_variables, points, angle,
              dic_for_other_variables, all_variables, device, plot_output_entry):
    '''Handels line plots w.r.t. a one dimensional variable.
    Inputs are the same as _plot().
    '''
    # First create the input dic. for the model
    plot_variable = plot_variables[0]
    domain_points, input_dic = _create_input_points(points, dic_for_other_variables,
                                                    all_variables, device,
                                                    plot_variable)
    output = _evaluate_model(model, solution_name, plot_output_entry, input_dic)
    output = _take_norm_of_output(plot_output_entry, output)
    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.grid()
    ax.plot(domain_points.flatten(), output.flatten())
    # add a text box for the values of the other variables
    if dic_for_other_variables is not None:
        info_string = _create_info_text(dic_for_other_variables)
        ax.text(1.05, 0.5, info_string, bbox={'facecolor': 'w', 'pad': 5},
                transform=ax.transAxes,)
    ax.set_xlabel(plot_variable.name)
    #plt.show()
    return fig


def surface2D_2_variables(model, solution_name, plot_variables, points, angle,
                          dic_for_other_variables,
                          all_variables, device, plot_output_entry):
    '''Handels surface plots, if the inputs are two intervals
    '''
    variable_1 = plot_variables[0]
    variable_2 = plot_variables[1]
    # Create the input dic.
    points = int(np.ceil(np.sqrt(points)))
    domain_1 = variable_1.domain.grid_for_plots(points)
    domain_2 = variable_2.domain.grid_for_plots(points)
    axis_1, axis_2 = np.meshgrid(domain_1, domain_2)
    input_dic = {variable_1.name: torch.FloatTensor(np.ravel(axis_1),
                                                    device=device).reshape(-1, 1),
                 variable_2.name: torch.FloatTensor(np.ravel(axis_2),
                                                    device=device).reshape(-1, 1)}
    input_dic = _create_input_dic(input_dic, points**2, dic_for_other_variables,
                                  all_variables, device)

    output = _evaluate_model(model, solution_name, plot_output_entry, input_dic)
    # Create the plot (we dont need a triangulation, since the plot over two intervalls
    # will be a rectangle)
    fig, ax = _create_figure_and_axis(angle)
    surf = ax.plot_surface(axis_1, axis_2, output.reshape(axis_1.shape),
                           cmap=cm.jet, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.4, aspect=5, pad=0.1)
    _add_textbox(dic_for_other_variables, ax, 1.2, 0.1)
    ax.set_xlabel(variable_1.name)
    ax.set_ylabel(variable_2.name)
    #plt.show()
    return fig


def _plot_for_two_outputs(plot_variables):
    '''Handles plots if two outputs of the model should be plotted.
    '''
    # plot a curve in 3D
    if len(plot_variables) == 1 and plot_variables[0].domain.dim == 1:
        return curve3D
    # plot vector field/quiver in 2D
    if len(plot_variables) == 1 and plot_variables[0].domain.dim == 2:
        return quiver2D
    else:
        raise NotImplementedError("""Can't plot 2D-output on given domains""")


def curve3D(model, solution_name, plot_variables, points, angle,
            dic_for_other_variables, all_variables, device, plot_output_entry):
    '''Handles curve plots where the output is 2D and the domain is 1D.
    '''
    # First create the input dic. for the model
    plot_variable = plot_variables[0]
    domain_points, input_dic = _create_input_points(points, dic_for_other_variables,
                                                    all_variables, device,
                                                    plot_variable)
    
    output = _evaluate_model(model, solution_name, plot_output_entry, input_dic)
    # Create the plot
    fig, ax = _create_figure_and_axis(angle)
    # Since we can't set the domain-axis in the center
    # (https://stackoverflow.com/questions/48442713/move-spines-in-matplotlib-3d-plot)
    # we plot a helper line to better show the structure of the curve
    domain_points = domain_points.flatten()
    ax.plot(domain_points, np.zeros_like(domain_points), np.zeros_like(domain_points),
            color='black') 
    # Now plot the curve
    ax.plot(domain_points, output[:, 0], output[:, 1])
    # add a text box for the values of the other variables
    _add_textbox(dic_for_other_variables, ax, 1.2, 0.1)
    ax.set_xlabel(plot_variable.name)
    return fig


def quiver2D(model, solution_name, plot_variables, points, angle,
             dic_for_other_variables, all_variables, device, plot_output_entry):
    '''Handles quiver/vector field plots w.r.t. a two dimensional variable.
    Inputs are the same as _plot().
    '''
    # First create the input dic. for the model
    plot_variable = plot_variables[0]
    domain_points, input_dic = _create_input_points(points, dic_for_other_variables,
                                                    all_variables, device,
                                                    plot_variable)

    output = _evaluate_model(model, solution_name, plot_output_entry, input_dic)
    # for the colors
    color = np.linalg.norm(output, axis=1)
    norm = colors.Normalize()
    #scale the arrows
    max_norm = np.max(color)
    output /= max_norm
    norm.autoscale(color)
    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.grid()
    _set_x_y_axis_data(plot_variable, ax)
    _outline_domain(plot_variable, ax)
    # create arrows
    ax.quiver(domain_points[:, 0], domain_points[:, 1], output[:, 0], output[:, 1], 
              color=cm.jet(norm(color)),
              units='xy', zorder=10)
    sm = cm.ScalarMappable(cmap=cm.jet, norm=norm)
    plt.colorbar(sm)
    # add a text box for the values of the other variables
    if dic_for_other_variables is not None:
        info_string = _create_info_text(dic_for_other_variables)
        ax.text(1.25, 0.5, info_string, bbox={'facecolor': 'w', 'pad': 5},
                transform=ax.transAxes)
    #plt.show()
    return fig


def contour_2D(model, solution_name, plot_variables, points, angle,
               dic_for_other_variables, all_variables, device, plot_output_entry):
    '''Handles colormap/contour plots w.r.t. a two dimensional variable.
    Inputs are the same as _plot().
    '''
    # First create the input dic. for the model
    plot_variable = plot_variables[0]
    domain_points, input_dic = _create_input_points(points, dic_for_other_variables,
                                                    all_variables, device,
                                                    plot_variable)

    output = _evaluate_model(model, solution_name, plot_output_entry, input_dic)
    output = _take_norm_of_output(plot_output_entry, output)
    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot()
    _set_x_y_axis_data(plot_variable, ax)
    ax.grid()
    _outline_domain(plot_variable, ax)
    # For complex domains it is best to triangulate them
    triangulation = _triangulation_of_domain(plot_variable, domain_points)
    cs = ax.tricontourf(triangulation, output.flatten(), 100, cmap=cm.jet)
    plt.colorbar(cs) 
    # add a text box for the values of the other variables
    if dic_for_other_variables is not None:
        info_string = _create_info_text(dic_for_other_variables)
        ax.text(1.3, 0.5, info_string, bbox={'facecolor': 'w', 'pad': 5},
                transform=ax.transAxes)
    #plt.show()
    return fig


def _create_input_points(points, dic_for_other_variables, all_variables,
                         device, plot_variable):
    """Creates the input of the model.
    """
    domain_points, input_dic = _create_domain(plot_variable, points, device)
    if plot_variable.domain.dim >= 2:
        points = len(domain_points)
    input_dic = _create_input_dic(input_dic, points, dic_for_other_variables,
                                  all_variables, device)
                                  
    return domain_points,input_dic


def _create_domain(plot_variable, points, device):
    domain_points = plot_variable.domain.grid_for_plots(points)
    input_dic = {plot_variable.name: torch.tensor(domain_points, device=device)}
    return domain_points, input_dic


def _create_input_dic(input_dic, points, dic_for_other_variables, all_variables,
                      device):
    '''Creates the input dictionary for the model

    Parameters
    ----------
    input_dic : dic
        The dictionary that already contains the data for the plot variable
    dic_for_other_variables : dic
        The data for all other variables
    all_variables : dic or list
        The right order of the dic    
    '''
    if dic_for_other_variables is not None: # create points for all other variables
        other_inputs = _create_dic_for_other_variables(
            points, dic_for_other_variables, device)
        input_dic = {**input_dic, **other_inputs}
    if all_variables is not None: # order the dic
        input_dic = _order_input_dic(input_dic, all_variables)
    return input_dic


def _create_dic_for_other_variables(points, dic_for_other_variables, device):
    '''Creates tensors for the other inputs of the model and puts them in 
    a dic.
    '''
    dic = {}
    for name in dic_for_other_variables:
        if isinstance(dic_for_other_variables[name], numbers.Number):
            dic[name] = dic_for_other_variables[name] * \
                torch.ones((points, 1), device=device)
        elif isinstance(dic_for_other_variables[name], (list, np.ndarray)):
            length = len(dic_for_other_variables[name])
            array = dic_for_other_variables[name]*np.ones((points, length))
            dic[name] = torch.FloatTensor(array, device=device)
        else:
            raise ValueError('Values for variables have to be a number or'
                             ' a list/array.')
    return dic


def _order_input_dic(input_dic, all_variables):
    order_dic = {}
    for vname in all_variables:
        order_dic[vname] = input_dic[vname]
    return order_dic


def _evaluate_model(model, solution_name, plot_output_entry, input_dic):
    # Evaluate the model
    output = model(input_dic)[solution_name]
    output = output.data.cpu().numpy()[:, plot_output_entry]
    return output


def _take_norm_of_output(plot_output_entry, output, axis=1):
    if len(plot_output_entry) > 1:
        # if we have many outputs take the norm
        output = np.linalg.norm(output, axis=axis)
    return output


def _add_textbox(dic_for_other_variables, ax, posi_x, posi_y):
    # add a text box for the values of the other variables
    if dic_for_other_variables is not None:
        info_string = _create_info_text(dic_for_other_variables)
        ax.text2D(posi_x, posi_y, info_string, bbox={'facecolor': 'w', 'pad': 5},
                  transform=ax.transAxes, ha="center")


def _create_info_text(dic_for_other_variables):
    info_text = ''
    for vname in dic_for_other_variables:
        info_text += vname + ' = ' + str(dic_for_other_variables[vname])
        info_text += '\n'
    return info_text[:-1]


def _scatter(plot_variables, data):
    """
    Create a scatter plot of given data points.

    Parameters
    ----------
    plot_variables : Variabale or list of Variables.
        The main variable(s) over which the points should be visualized.
    data : dict
        A dictionary that contains the points for every Variables.
    """
    axes = []
    for v in plot_variables:
        axes.extend(torch.chunk(data[v].detach().cpu(), data[v].shape[1], dim=1))
    labels = []
    for v in plot_variables:
        for _ in range(data[v].shape[1]):
            labels.append(v)

    fig = plt.figure()
    if len(axes) == 1:
        ax = fig.add_subplot()
        axes.append(torch.zeros_like(axes[0]))
        ax.set_xlabel(labels[0])
    elif len(axes) == 2:
        ax = fig.add_subplot()
        ax.grid()
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
    elif len(axes) == 3:
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
    else:
        raise NotImplementedError("Plot variables should be 1d, 2d or 3d.")
    ax.scatter(*axes)
    return fig


def _triangulation_of_domain(plot_variable, domain_points):
    points = np.vstack([domain_points[:, 0], domain_points[:, 1]]).T
    tess = scipy.spatial.Delaunay(points)  # create triangulation
    tri = np.empty((0, 3))
    # check what triangles are inside
    for t in tess.simplices:
        p = points[t]
        middle_point = 1/3.0 * (p[0] + p[1] + p[2])
        if plot_variable.domain.is_inside([middle_point]):
            tri = np.append(tri, [t], axis=0)

    return mtri.Triangulation(x=points[:, 0], y=points[:, 1], triangles=tri)


def _decide_plot_entries(model, solution_name, plot_output_entries):
    # set/decide the number of outputs which have to be plotted.
    if not isinstance(plot_output_entries, list):
        plot_output_entries = [plot_output_entries]
    if plot_output_entries[0] == -1:
        plot_output_entries = np.arange(0, 
                                        model.solution_dims[solution_name],
                                        dtype=int)
    return plot_output_entries


def _outline_domain(plot_variable, ax):
    """Outlines a 2D-domain
    """
    poly = plot_variable.domain.outline()
    if isinstance(poly, (patches.Rectangle, patches.Circle, patches.Polygon)):
        ax.add_patch(poly)
    else:  # domain operations or polynom are used:
        for p in poly:
            ax.plot(p[:, 0], p[:, 1], color='k', linewidth=2, linestyle='--')


def _set_x_y_axis_data(plot_variable, ax):
    # scale the border
    bounds = plot_variable.domain._compute_bounds()
    width = bounds[1] - bounds[0]
    height = bounds[3] - bounds[2]
    scale_x = 0.05*width
    scale_y = 0.05*height
    ax.set_xlim((bounds[0]-scale_x, bounds[1]+scale_x))
    ax.set_ylim((bounds[2]-scale_y, bounds[3]+scale_y))
    ax.set_xlabel(plot_variable.name + '_1')
    ax.set_ylabel(plot_variable.name + '_2')


def _create_figure_and_axis(angle):
    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(angle[0], angle[1])
    return fig,ax
"""
def _check_triangle_inside(triangle, domain):
    boundary_points = triangle.points_on_boundary(10)
    inside = domain.is_inside(boundary_points)
    boundary = domain.is_on_boundary(boundary_points)
    return np.logical_or(inside, boundary).sum() >= 7 # only check that some points are
                                                      # correct, because of numeric 
                                                      # errors


class __HelperTriangle():
    '''A helper class for triangulation of the domain.
    The Triangle class of domain2D computes, when initialized, things 
    that are not needed for the triangulation (e.g. inverse transformation matrix...)

    Parameters
    -----------
    corner_1, corner_2, corner_3 : array_like
        The three corners of the triangle.
    '''
    def __init__(self, corner_1, corner_2, corner_3):
        self.corners = np.array([corner_1, corner_2, corner_3, corner_1])
        self.side_lengths = Triangle._compute_side_lengths(self, self.corners)
        self.surface = sum(self.side_lengths)

    def points_on_boundary(self, n):
        line_points = np.linspace(0, self.surface, n+1)[:-1]
        return Triangle._distribute_line_to_boundary(self, line_points,
                                                     self.corners, self.side_lengths)
"""                                               