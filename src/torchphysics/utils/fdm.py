'''Implements a basic FDM

Right now really bad and pretty specific for the heat equation example
'''
import numpy as np

from .helper import apply_user_fun


def FDM(domain_dic, step_width_dic, time_interval, variable_list, initial_condition):
    solution = []
    time_domains = []
    dx = step_width_dic['x'][0]
    dy = step_width_dic['x'][1]
    domain = _create_domain(domain_dic['x'], step_width_dic['x'])
    for k in range(len(variable_list)):
        dt = dx**2 * dy**2 / (2 * variable_list[k] * (dx**2 + dy**2))
        step_width_dic['t'] = dt
        time = _create_domain([time_interval], [dt])
        u = _create_solution_array(domain, time)
        u[0, :, :] = _set_initial_condition(
            u[0, :, :], domain, time[0], initial_condition)
        for i in range(1, len(time[0])):
            u[i, :, :] = do_timestep(u[i-1, :, :], domain, time,
                                     step_width_dic, variable_list[k])
        solution.append(u)
        time_domains.append(time[0])
    return domain, time_domains, solution


def do_timestep(u0, domain, time, step_width_dic, variable):
    '''Implements the time step and fdm scheme of the methode  
    '''
    D = variable
    dt = step_width_dic['t']
    dx2 = step_width_dic['x'][0]**2
    dy2 = step_width_dic['x'][1]**2
    u = u0.copy()
    u[1:-1, 1:-1] = u0[1:-1, 1:-1] + D * dt * (
        (u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])/dx2
        + (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])/dy2)

    return u


def _create_domain(domain_list, step_width_list):
    domain = [[] for i in range(len(domain_list))]
    for dim in range(len(domain_list)):
        step_number = int((domain_list[dim][1] - domain_list[dim][0])
                          / step_width_list[dim])
        domain[dim][:] = np.linspace(domain_list[dim][0],
                                     domain_list[dim][1],
                                     step_number+1)
    return domain


def _create_solution_array(domain, time):
    array_dim = []
    array_dim.append(len(time[0]))
    for i in range(len(domain)):
        array_dim.append(len(domain[i]))

    solution_array = np.zeros(tuple(array_dim))
    return solution_array


def _set_initial_condition(u, domain, time, initial_condition):
    u0 = u.copy()
    dic = {'t': time}
    for i in range(len(domain[0])):
        for j in range(len(domain[1])):
            dic['x'] = np.array([domain[0][i], domain[1][j]]).reshape(-1, 2)
            u0[i, j] = apply_user_fun(initial_condition,
                                      dic,
                                      whole_batch=False,
                                      batch_size=1)[1]
    return u0


def create_validation_data(domain, time, u, D_list, D_is_input):
    '''
    This should become a more general version, where we put in a known function
    and all variables to get a dictionary for data conditions
    '''
    points = np.empty((0, 4))
    data_u = np.empty((0, 1))
    for k in range(len(D_list)):
        new_points = np.array(np.meshgrid(
            domain[0], domain[1], time[k], D_list[k])).T.reshape(-1, 4)
        new_data_u = u[k].reshape(-1, 1)
        points = np.concatenate((points, new_points)).astype(np.float32)
        data_u = np.concatenate((data_u, new_data_u)).astype(np.float32)
    data_x = {'x': points[:, [0, 1]], 't': points[:, [2]]}
    if D_is_input:
        data_x['D'] = points[:, [3]]
    return data_x, data_u
