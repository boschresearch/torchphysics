'''This file contains different functions for plotting outputs of
neural networks
'''
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import scipy.spatial
import numpy as np
import torch
from matplotlib import cm, colors

from ..user_fun import UserFunction
from ...problem.spaces.points import Points


class Plotter():
    '''Object to collect plotting properties.

    Parameters
    ----------
    plot_function : callable
        A function that specfices the part of the model that should be plotted.
        Can be of the same form as the condition-functions.
        E.g. if the solution name is 'u' we can use

        |    plot_func(u):
        |        return u[:, 0]

        to plot the first entry of 'u'. For the derivative we could write:

        |    plot_func(u, x):
        |        return grad(u, x)
        
    point_sampler : torchphysics.samplers.PlotSampler
        A Sampler that creates the points that should be used for the
        plot.
    angle : list, optional
        The view angle for surface plots. Standart angle is [30, 30]
    log_interval : int
        Plots will be saved every log_interval steps if the plotter is used in
        training of a model.
    plot_type : str, optional
        Specifies how the output should be plotted. If no input is given, the method
        will try to use a fitting way, to show the data. Implemented types are:
            - 'line' for plots in 1D
            - 'surface_2D' for surface plots, with a 2D-domain
            - 'curve' for a curve in 3D, with a 1D-domain, 
            - 'quiver_2D' for quiver/vector field plots, with a 2D-domain
            - 'contour_surface' for contour/colormaps, with a 2D-domain
    kwargs:
        Additional arguments to specify different parameters/behaviour of
        the plot. See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html
        for possible arguments of each underlying object.
    '''

    def __init__(self, plot_function, point_sampler, angle=[30, 30],
                 log_interval=None, plot_type='', **kwargs):
        self.plot_function = UserFunction(plot_function)
        self.point_sampler = point_sampler
        self.angle = angle
        self.log_interval = log_interval
        self.plot_type = plot_type
        self.kwargs = kwargs

    def plot(self, model):
        """Creates the plot of the model.

        Parameters
        ----------
        model : torchphysics.models.Model
            The Model/neural network that should be used in the plot.  

        Returns
        -------
        plt.figure
            The figure handle of the created plot     
        """
        return plot(model=model, plot_function=self.plot_function,
                    point_sampler=self.point_sampler, 
                    angle=self.angle, plot_type=self.plot_type, **self.kwargs)


def plot(model, plot_function, point_sampler, angle=[30, 30], plot_type='', 
         device='cpu', **kwargs):
    '''Main function for plotting

    Parameters
    ----------
    model : torchphysics.models.Model
        The Model/neural network that should be used in the plot.
    plot_function : callable
        A function that specfices the part of the model that should be plotted.
        Of the same form as the condition-functions.
        E.g. if the solution name is 'u', we can use

        |    plot_func(u):
        |        return u[:, 0]

        to plot the first entry of 'u'. For the derivative we could write:

        |    plot_func(u, x):
        |        return grad(u, x)

    point_sampler : torchphysics.samplers.PlotSampler
        A Sampler that creates the points that should be used for the
        plot.
    angle : list, optional
        The view angle for 3D plots. Standard angle is [30, 30]
    plot_type : str, optional
        Specifies how the output sholud be plotted. If no input is given the method
        will try to use a fitting way to show the data. Implemented types are:
            - 'line' for plots in 1D
            - 'surface_2D' for surface plots, with a 2D-domain
            - 'curve' for a curve in 3D, with a 1D-domain, 
            - 'quiver_2D' for quiver/vector-field plots, with a 2D-domain
            - 'contour_surface' for contour/colormaps, with a 2D-domain
    kwargs:
        Additional arguments to specify different parameters/behaviour of
        the plot. See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html
        for possible arguments of each underlying object.

    Returns
    -------
    plt.figure
        The figure handle of the created plot


    Notes
    -----
    What this function does is:
    creating points with sampler -> evaluate model -> evalute plot function
    -> create the plot with matplotlib.pyplot.

    The function is only meant to  give a fast overview over the trained neural network.
    In general the methode is not optimized for complex domains.
    '''
    if not isinstance(plot_function, UserFunction):
        plot_function = UserFunction(fun=plot_function)
    inp_points, output, out_shape = _create_plot_output(model, plot_function,
                                                        point_sampler, device)
    domain_points = _extract_domain_points(inp_points, point_sampler.domain, 
                                           len(point_sampler))
    plot_fun = _find_plot_function(point_sampler, out_shape, plot_type)
    if plot_fun is not None:
        return plot_fun(output=output, domain_points=domain_points,
                        point_sampler=point_sampler, angle=angle, **kwargs)
    else:
        raise NotImplementedError(f"""Plotting for a {out_shape[1]} 
                                      dimensional output is not implemented!  
                                      Please specify the output to plot.""")


def _create_plot_output(model, plot_function, point_sampler, device):
    # first create the plot points and evaluate the model
    inp_points = point_sampler.sample_points(device=device)
    inp_points_dict = inp_points.coordinates
    model_out = model(Points.from_coordinates(inp_points_dict))
    data_dict = {**model_out.coordinates, **inp_points_dict}
    # When model output correct: data_dict = {**model_out.coordinates, **inp_points.coordinates}
    # now evaluate the plot function
    output = plot_function(data_dict)
    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()
    # get dimension of the output
    out_shape = _compute_output_shape(output)
    return inp_points, output, out_shape


def _compute_output_shape(output):
    out_shape = 1
    if len(np.shape(output)) > 1:
        out_shape = np.shape(output)[1]     
    return out_shape


def _extract_domain_points(input_points, domain, length):
    # for the plot, the points need to be in a array not a dictionary
    # and are not allowed to be a tensor
    domain_points = np.zeros((length, domain.dim))
    current_dim = 0
    for vname in domain.space:
        v_dim = domain.space[vname]
        plot_points = input_points[:, [vname]].as_tensor
        domain_points[:, current_dim:current_dim+v_dim] = \
            plot_points.detach().cpu().numpy()
        current_dim += v_dim
    return domain_points


def _find_plot_function(point_sampler, out_shape, plot_type):
    # check if a plot type is specified
    plot_types = {'line': line_plot, 'surface_2D': surface2D,
                  'curve': curve3D, 'quiver_2D': quiver2D,
                  'contour_surface': contour_2D}
    plot_fun = plot_types.get(plot_type)
    # If no (or wrong) type is given, try to find the correct type:
    if plot_fun is None:
        if out_shape == 1:
            plot_fun = _plot_for_one_output(point_sampler.domain.dim)
        if out_shape == 2:
            plot_fun = _plot_for_two_outputs(point_sampler.domain.dim)
    return plot_fun


def _plot_for_one_output(domain_dim):
    '''Handles plots if only one output of the model should be plotted.
    '''
    # surface plots:
    if domain_dim == 2:
        return surface2D
    # line plots:                       
    elif domain_dim == 1:
        return line_plot
    else:
        raise NotImplementedError("""Can't plot 1D-output on given domain""")


def _plot_for_two_outputs(domain_dim):
    '''Handles plots if two outputs of the model should be plotted.
    '''
    # plot a curve in 3D
    if domain_dim == 1:
        return curve3D
    # plot vector field/quiver in 2D
    elif domain_dim == 2:
        return quiver2D
    else:
        raise NotImplementedError("""Can't plot 2D-output on given domain""")


def surface2D(output, domain_points, point_sampler, angle, **kwargs):
    '''Handels surface plots w.r.t. a two dimensional variable.
    '''
    # For complex domains it is best to triangulate them for the plotting
    triangulation = _triangulation_of_domain(point_sampler.domain, domain_points)
    fig, ax = _create_figure_and_axis(angle)
    _set_x_y_axis_data(point_sampler, ax)
    if not 'antialiased' in kwargs:
        kwargs['antialiased'] = False
    if not 'linewidth' in kwargs:
        kwargs['linewidth'] = 0
    surf = ax.plot_trisurf(triangulation, output.flatten(),
                           cmap=cm.jet, **kwargs)
    fig.colorbar(surf, shrink=0.4, aspect=5, pad=0.1)
    _add_textbox(point_sampler.data_for_other_variables, ax, 1.2, 0.1)
    return fig


def line_plot(output, domain_points, point_sampler, angle, **kwargs):
    '''Handels line plots w.r.t. a one dimensional variable.
    '''
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.grid()
    if len(output.shape) > 1 and output.shape[1] > 1:
        raise ValueError("""Can't plot a line with a multidimensional output. 
                            If u want to plot the norm use: torch.linalg.norm inside 
                            the plot function.""")
    ax.plot(domain_points.flatten(), output.flatten(), **kwargs)
    # add a text box for the values of the other variables
    if len(point_sampler.data_for_other_variables) > 0:
        info_string = _create_info_text(point_sampler.data_for_other_variables)
        ax.text(1.05, 0.5, info_string, bbox={'facecolor': 'w', 'pad': 5},
                transform=ax.transAxes,)
    ax.set_xlabel(list(point_sampler.domain.space.keys())[0])
    return fig


def curve3D(output, domain_points, point_sampler, angle, **kwargs):
    '''Handles curve plots where the output is 2D and the domain is 1D.
    '''
    fig, ax = _create_figure_and_axis(angle)
    # Since we can't set the domain-axis in the center
    # (https://stackoverflow.com/questions/48442713/move-spines-in-matplotlib-3d-plot)
    # we plot a helper line to better show the structure of the curve
    domain_points = domain_points.flatten()
    ax.plot(domain_points, np.zeros_like(domain_points), np.zeros_like(domain_points),
            **kwargs) 
    # Now plot the curve
    ax.plot(domain_points, output[:, 0], output[:, 1])
    # add a text box for the values of the other variables
    _add_textbox(point_sampler.data_for_other_variables, ax, 1.2, 0.1)
    ax.set_xlabel(list(point_sampler.domain.space.keys())[0])
    return fig


def quiver2D(output, domain_points, point_sampler, angle, **kwargs):
    '''Handles quiver/vector field plots w.r.t. a two dimensional variable.
    '''
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
    _set_x_y_axis_data(point_sampler, ax)
    # create arrows
    ax.quiver(domain_points[:, 0], domain_points[:, 1], output[:, 0], output[:, 1], 
              color=cm.jet(norm(color)),
              units='xy', zorder=10, **kwargs)
    sm = cm.ScalarMappable(cmap=cm.jet, norm=norm)
    plt.colorbar(sm, ax=ax)
    # add a text box for the values of the other variables
    if len(point_sampler.data_for_other_variables) > 0:
        info_string = _create_info_text(point_sampler.data_for_other_variables)
        ax.text(1.25, 0.5, info_string, bbox={'facecolor': 'w', 'pad': 5},
                transform=ax.transAxes)
    return fig


def contour_2D(output, domain_points, point_sampler, angle, **kwargs):
    '''Handles colormap/contour plots w.r.t. a two dimensional variable.
    '''
    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot()
    _set_x_y_axis_data(point_sampler, ax)
    ax.grid()
    # For complex domains it is best to triangulate them
    triangulation = _triangulation_of_domain(point_sampler.domain,
                                             domain_points)
    if len(output.shape) > 1 and output.shape[1] > 1:
        raise ValueError("""Can't plot a surface with a multidimensional output. 
                            If u want to plot the norm use: torch.linalg.norm inside 
                            the plot function.""")
    if not 'levels' in kwargs:
        kwargs['levels'] = 100
    cs = ax.tricontourf(triangulation, output.flatten(), cmap=cm.jet, **kwargs)
    plt.colorbar(cs) 
    # add a text box for the values of the other variables
    if len(point_sampler.data_for_other_variables) > 0:
        info_string = _create_info_text(point_sampler.data_for_other_variables)
        ax.text(1.3, 0.5, info_string, bbox={'facecolor': 'w', 'pad': 5},
                transform=ax.transAxes)
    return fig


def _add_textbox(data_for_other_variables, ax, posi_x, posi_y):
    # add a text box for the values of the other variables
    if len(data_for_other_variables) > 0:
        info_string = _create_info_text(data_for_other_variables)
        ax.text2D(posi_x, posi_y, info_string, bbox={'facecolor': 'w', 'pad': 5},
                  transform=ax.transAxes, ha="center")


def _create_info_text(data_for_other_variables):
    info_text = ''
    data_dict = data_for_other_variables.coordinates
    for vname, data in data_dict.items():
        if data.shape[1] == 1:
            data = data.item()
        else:
            data = data[0].detach().cpu().numpy()
        if isinstance(data, float):
            data = round(data, 4)
        info_text += vname + ' = ' + str(data)
        info_text += '\n'
    return info_text[:-1]


def _scatter(plot_variables, data):
    """
    Create a scatter plot of given data points.

    Parameters
    ----------
    plot_variables : dict
        The main variable(s) for which the points should be visualized.
    data : torchphysics.problem.spaces.Points
        The points that contain the values for every Variable.
    """
    axes = []
    labels = []
    for v in plot_variables.keys():
        # get axis dimension (type of plot) and set points
        data_for_v = data[:, [v]].as_tensor
        axes.extend(torch.chunk(data_for_v.detach().cpu(),
                    data_for_v.shape[1], dim=1))
        # set label names
        for _ in range(data_for_v.shape[1]):
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


def _triangulation_of_domain(domain, domain_points):
    points = np.vstack([domain_points[:, 0], domain_points[:, 1]]).T
    tess = scipy.spatial.Delaunay(points)  # create triangulation
    tri = np.empty((0, 3))
    # check what triangles are inside
    for t in tess.simplices:
        p = points[t]
        center = 1/3.0 * (p[0] + p[1] + p[2])
        embed_point = Points(torch.tensor([center]), domain.space)
        if domain.__contains__(embed_point):
            tri = np.append(tri, [t], axis=0)

    return mtri.Triangulation(x=points[:, 0], y=points[:, 1], triangles=tri)


def _set_x_y_axis_data(point_sampler, ax):
    # scale the border
    bounds = point_sampler.domain.bounding_box()
    width = bounds[1] - bounds[0]
    height = bounds[3] - bounds[2]
    scale_x = 0.05*width
    scale_y = 0.05*height
    ax.set_xlim((bounds[0]-scale_x, bounds[1]+scale_x))
    ax.set_ylim((bounds[2]-scale_y, bounds[3]+scale_y))
    vname = list(point_sampler.domain.space.keys())
    if len(vname) == 1:
        ax.set_xlabel(vname[0] + '_1')
        ax.set_ylabel(vname[0] + '_2')
    else:
        ax.set_xlabel(vname[0])
        ax.set_ylabel(vname[1])      


def _create_figure_and_axis(angle):
    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(angle[0], angle[1])
    return fig,ax  