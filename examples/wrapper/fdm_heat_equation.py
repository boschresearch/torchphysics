'''Implements a basic FDM to construct validation data for the heat equation. 
'''
import numpy as np
import torch
from torchphysics.utils import UserFunction
from torchphysics.problem.spaces import Points


def FDM(domain_bounds, step_width_list, time_interval, parameter_list,
        initial_condition):
    """A simple FDM with a explicit euler to create some validation/inverse 
    data for the heatequation examples. Only works for rectangular domains and
    Dirichlet boundary conditions.

    Parameters
    ----------
    domain_bounds : list
        The bounds of the rectangular domain, in the form:
        [min_x1, max_x1, min_x2, max_x2]
    step_width_list : list
        The step size for the domain disrectization, in the form:
        [step_x1, step_x2]
    time_interval: list
        The time interval for which the solution should be computed.
    parameter_list : list
        The different parameters for the heat equation, will solve the 
        equation for each params in the list and append the solutions.
    initial_condition: callable
        The initail condition.

    Returns
    -------
    domain : list
        The list of domain points. The first entry contains the x1-axis, the
        sconde entry the x2-axis. For plotting the meshgrid of the points
        have to be created.
    time_domains : list
        A list containing the different time domains.
        Since the time step size scales with the used coefficent for the heat
        equation, the number of time points may vary.
        The i-th entry in the list corresspond to the i-th coefficent.
    solutions : list
        The different solutions, in the same order as time_domains.

    Notes
    -----
    Only can solve the eq: du/dt = D*laplcian(u).
    """
    solution = []
    time_domains = []
    dx = step_width_list[0]
    dy = step_width_list[1]
    initial_condition = UserFunction(initial_condition)
    domain = _create_domain(domain_bounds, step_width_list)
    # solve for each coefficent the heat equation
    for k in parameter_list:
        # scale time step to make scheme stable 
        dt = dx**2 * dy**2 / (2 * k * (dx**2 + dy**2))
        time = _create_domain(time_interval, [dt])
        u = _create_solution(domain, time)
        _set_initial_condition(u[0, :, :], domain, time[0], initial_condition)
        for i in range(1, len(time[0])):
            u[i, :, :] = do_timestep(u[i-1, :, :], dx, dy, dt, k)
        solution.append(u)
        time_domains.append(time[0])
    return domain, time_domains, solution


def do_timestep(u0, dx, dy, dt, variable):
    '''Implements the time step and fdm scheme of the methode  
    '''
    D = variable
    dx2 = dx**2
    dy2 = dy**2
    u = u0.detach().clone()
    u[1:-1, 1:-1] = u0[1:-1, 1:-1] + D * dt * (
        (u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])/dx2
        + (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])/dy2)

    return u


def _create_domain(domain_list, step_width_list):
    domain = [[] for _ in range(int(len(domain_list)/2))]
    for dim in range(len(domain)):
        step_number = int((domain_list[2*dim+1] - domain_list[2*dim])
                          / step_width_list[dim])
        domain[dim][:] = torch.linspace(domain_list[2*dim], domain_list[2*dim+1],
                                        step_number+1)
    return domain


def _create_solution(domain, time):
    # creates the 'empyt' tensor to later store the solution
    array_dim = []
    array_dim.append(len(time[0]))
    for i in range(len(domain)):
        array_dim.append(len(domain[i]))
    solution_array = torch.zeros(tuple(array_dim))
    return solution_array


def _set_initial_condition(u0, domain, time, initial_condition):
    input_dict = {'t': time[0]}
    for i in range(len(domain[0])):
        for j in range(len(domain[1])):
            input_dict['x'] = torch.tensor([domain[0][i], domain[1][j]]).reshape(-1, 2)
            u0[i, j] = initial_condition(input_dict)


def transform_to_points(domain, time_list, solution_list, parameter_list,
                        param_is_input):
    """Transforms the FDM-data to data that can be used by the conditions.
    Only meant to be used for the heat equation example.
    
    Parameters
    ----------
    domain : list
        The domain list from the FDM.
    time_list : list
        The list of time points from the FDM.
    soluion_list : list
        The list of solutions from the FDM.
    parameter_list : list
        The list of used parameters.
    param_is_input : bool
        If the parameter is a input variable of the model.


    Returns
    -------
    inp_data : Points
        The input points for the model.
    out_data : Poitns
        The expected output.
    """
    points = torch.empty((0, 4))
    out_data = torch.empty((0, 1))
    for k in range(len(parameter_list)):
        new_points = np.array(np.meshgrid(
            domain[0], domain[1], time_list[k], parameter_list[k])).T.reshape(-1, 4)
        new_points = torch.as_tensor(new_points)
        new_data_u = solution_list[k].reshape(-1, 1)
        points = torch.cat((points, new_points)).float()
        out_data = torch.cat((out_data, new_data_u)).float()
    inp_data = {'x': points[:, [0, 1]], 't': points[:, [2]]}
    out_data = {'u': out_data}
    if param_is_input:
        inp_data['D'] = points[:, [3]]
    return Points.from_coordinates(inp_data), Points.from_coordinates(out_data)
