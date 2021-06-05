'''This file contains different functions for plotting outputs of 
neural networks
'''
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch
import numbers


class Plotter():
    '''Object to collect plotting properties

    Parameters
    ----------
    plot_variables : Variabale or list of Variables.
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
    '''
    def __init__(self, plot_variables, points,
         dic_for_other_variables=None, all_variables=None):
        self.plot_variables = plot_variables
        self.points = points
        self.dic_for_other_variables = dic_for_other_variables
        self.all_variables = all_variables

    def plot(self, model, device='cpu'):
        return _plot(model, self.plot_variables, self.points, self.dic_for_other_variables,
                    self.all_variables, device=device)
        

def _plot(model, plot_variables, points,
         dic_for_other_variables=None, all_variables=None, device='cpu'):
    '''Main function for plotting

    Parameters
    ----------
    model : DiffEqModel
        A neural network of which the output should be plotted
    plot_variables : Variabale or list of Variables.
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
    device : str or torch device
        The device of the model.

    Returns
    -------
    plt.figure
        The figure handle of the created plot     
    '''
    if not isinstance(plot_variables, list):
        plot_variables = [plot_variables]
    if len(plot_variables) == 1 and plot_variables[0].domain.dim == 2:
        return _plot2D(model, plot_variables[0], points,
                       dic_for_other_variables, all_variables, device)
    elif len(plot_variables) == 1 and plot_variables[0].domain.dim == 1:
        return _plot1D(model, plot_variables[0], points,
                       dic_for_other_variables, all_variables, device)
    elif (len(plot_variables) == 2 and
          plot_variables[0].domain.dim + plot_variables[0].domain.dim == 2):
        return _plot2D_2_variables(model, plot_variables[0], plot_variables[1],
                                   points, dic_for_other_variables,
                                   all_variables, device)
    else:
        raise NotImplementedError


def _plot2D(model, plot_variable, points, dic_for_other_variables, all_variables, device):
    domain_points, input_dic = _create_domain(plot_variable, points, device)
    points = len(domain_points)
    input_dic = _create_input_dic(input_dic, points, dic_for_other_variables,
                                  all_variables, device)
    output = model.forward(input_dic, track_gradients=False).data.cpu().numpy()
    axis_1 = domain_points[:, 0]
    axis_2 = domain_points[:, 1]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # Not perfect for concave domains, see:
    # https://stackoverflow.com/questions/9170838/surface-plots-in-matplotlib
    # Should be added if we get problems of this type
    surf = ax.plot_trisurf(axis_1, axis_2, output.flatten(),
                           cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.4, aspect=5, pad=0.1)
    if dic_for_other_variables is not None:
        info_string = _create_info_text(dic_for_other_variables)
        ax.text2D(1.2, 0.1, info_string, bbox={'facecolor': 'w', 'pad': 5},
                  transform=ax.transAxes, ha="center")

    ax.set_xlabel(plot_variable.name + '_1')
    ax.set_ylabel(plot_variable.name + '_2')
    plt.show()
    return fig


def _plot1D(model, plot_variable, points, dic_for_other_variables, all_variables, device):
    domain_points, input_dic = _create_domain(plot_variable, points, device)
    input_dic = _create_input_dic(input_dic, points, dic_for_other_variables,
                                  all_variables, device)
    output = model.forward(input_dic, track_gradients=False).data.cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.grid()
    ax.plot(domain_points.flatten(), output.flatten())
    if dic_for_other_variables is not None:
        info_string = _create_info_text(dic_for_other_variables)
        ax.text(1.05, 0.5, info_string, bbox={'facecolor': 'w', 'pad': 5},
                transform=ax.transAxes,)
    ax.set_xlabel(plot_variable.name)
    plt.show()
    plt.ion()
    return fig


def _plot2D_2_variables(model, variable_1, variable_2, points,
                        dic_for_other_variables, all_variables, device):

    points = int(np.ceil(np.sqrt(points)))
    domain_1 = variable_1.domain.grid_for_plots(points)
    domain_2 = variable_2.domain.grid_for_plots(points)
    axis_1, axis_2 = np.meshgrid(domain_1, domain_2)
    input_dic = {variable_1.name: torch.FloatTensor(np.ravel(axis_1, device=device).reshape(-1, 1)),
                 variable_2.name: torch.FloatTensor(np.ravel(axis_2, device=device).reshape(-1, 1))}
    input_dic = _create_input_dic(input_dic, points**2, dic_for_other_variables,
                                  all_variables, device)
    output = model.forward(input_dic, track_gradients=False).data.cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(axis_1, axis_2, output.reshape(axis_1.shape),
                           cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.4, aspect=5, pad=0.1)
    if dic_for_other_variables is not None:
        info_string = _create_info_text(dic_for_other_variables)
        ax.text2D(1.2, 0.1, info_string, bbox={'facecolor': 'w', 'pad': 5},
                  transform=ax.transAxes, ha="center")

    ax.set_xlabel(variable_1.name)
    ax.set_ylabel(variable_2.name)
    plt.show()
    plt.ion()
    return fig


def _create_domain(plot_variable, points, device):
    domain_points = plot_variable.domain.grid_for_plots(points)
    input_dic = {plot_variable.name: torch.tensor(domain_points, device=device)}
    return domain_points, input_dic


def _create_input_dic(input_dic, points, dic_for_other_variables, all_variables, device):
    if dic_for_other_variables is not None:
        other_inputs = _create_dic_for_other_variables(points, dic_for_other_variables, device)
        input_dic = input_dic | other_inputs
    if all_variables is not None:
        input_dic = _order_input_dic(input_dic, all_variables)
    return input_dic


def _create_dic_for_other_variables(points, dic_for_other_variables, device):
    dic = {}
    for name in dic_for_other_variables:
        if isinstance(dic_for_other_variables[name], numbers.Number):
            dic[name] = dic_for_other_variables[name]*torch.ones((points, 1), device=device)
        elif isinstance(dic_for_other_variables[name], list):
            length = len(dic_for_other_variables[name])
            array = dic_for_other_variables[name]*np.ones((points, length))
            dic[name] = torch.FloatTensor(array, device=device)
        else:
            raise ValueError('Values for variables have to be numbers or lists/arrays.')
    return dic


def _order_input_dic(input_dic, all_variables):
    order_dic = {}
    for vname in all_variables:
        order_dic[vname] = input_dic[vname]
    return order_dic


def _create_info_text(dic_for_other_variables):
    info_text = ''
    for vname in dic_for_other_variables:
        info_text += vname + ' = ' + str(dic_for_other_variables[vname])
        info_text += '\n'
    return info_text[:-1]
