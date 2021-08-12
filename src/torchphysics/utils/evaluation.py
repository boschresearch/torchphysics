'''File contains different helper functions to get specific informations about
the computed solution.
'''
import time
import torch
import numpy as np
from . import plot as plt


def get_min_max_inside(model, solution_name, domain_variable, resolution, device='cpu',
                       dic_for_other_variables=None, all_variables=None):
    '''Computes the minimum and maximum values of the model w.r.t. the given
    variables.

    Parameters
    ----------
    model : DiffEqModel
        A neural network of which values should be computed.
    solution_name : str
        The output function for which the min. and max. should be computed.
    domain_variable : Variabale
        The main variable(s) over which the solution should be evaluated. For
        this variable a grid will be put over the domain. 
    resolution : int
        The number of points that should be used.
    device : str or torch device
        The device of the model.    
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

    Returns
    -------
    float
        The minimum of the model.
    float 
        The maximum  of the model.
    '''
    print('-- Start evaluation of minimum and maximum --')
    domain_points = domain_variable.domain._grid_sampling_inside(resolution)
    input_dic = {domain_variable.name : torch.tensor(domain_points, device=device)}
    input_dic = plt._create_input_dic(input_dic, resolution, dic_for_other_variables,
                                      all_variables, device)
 
    start = time.time()
    pred = model(input_dic)
    end = time.time()
    pred = pred[solution_name].data.cpu().numpy()
    max_pred = np.max(pred)
    min_pred = np.min(pred)
    print('Time to evaluate model:', end - start)
    print('For the variables:', dic_for_other_variables)
    print('Found inside:')
    print('Max:', max_pred)
    print('Min:', min_pred)
    return min_pred, max_pred


def get_min_max_boundary(model, solution_name, boundary_variable, resolution,
                         device='cpu', dic_for_other_variables=None,
                         all_variables=None):
    '''Computes the minimum and maximum values of the model w.r.t. the given
    variables (at the boundary).

    Parameters
    ----------
    model : DiffEqModel
        A neural network of which values should be computed.
    solution_name : str
        The output function for which the min. and max. should be computed.
    boundary_variable : Variabale
        The main variable(s) over which the solution should be evaluated. For
        this variable a grid will be put over the boundary. 
    resolution : int
        The number of points that should be used.
    device : str or torch device
        The device of the model.    
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

    Returns
    -------
    float
        The minimum of the model.
    float 
        The maximum  of the model.
    '''
    print('-- Start evaluation of minimum and maximum at the boundary --')
    domain_points = boundary_variable.domain._grid_sampling_boundary(resolution)
    input_dic = {boundary_variable.name : torch.tensor(domain_points, device=device)}
    input_dic = plt._create_input_dic(input_dic, resolution, dic_for_other_variables,
                                      all_variables, device) 
    start = time.time()
    pred = model(input_dic)
    end = time.time()
    pred = pred[solution_name].data.cpu().numpy()
    max_pred = np.max(pred)
    min_pred = np.min(pred)
    print('Time to evaluate model:', end - start)
    print('For the variables:', dic_for_other_variables)
    print('Found at the boundary:')
    print('Max:', max_pred)
    print('Min:', min_pred)
    return min_pred, max_pred