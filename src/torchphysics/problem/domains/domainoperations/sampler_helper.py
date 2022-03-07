"""This file contains some sample functions for the domain operations.
Since Union/Cut/Intersection follow the same idea for sampling for a given number of points.
"""
import torch
import warnings

from torchphysics.problem.spaces.points import Points


def _inside_random_with_n(main_domain, domain_a, domain_b, n, params, invert, device):
    """Creates a random uniform points inside of a cut or intersection domain.

    Parameters
    ----------
    main_domain : Domain
        The domain that represents the new created domain.
    domain_a, domain_b : Domain
        The two domains that define the main domain.
    n : int
        The number of points.
    params : Points
        Additional parameters for the domains.
    invert : bool
        Says if the points should lay in the domain_b (intersection) or if
        not (cut). For the Cut-Domain it is invert=True.
    device : str
        The device on which the points should be created.
    """
    if n == 1: 
        return _random_points_if_n_eq_1(main_domain, domain_a, domain_b,
                                        params, invert, device)
    return _random_points_inside(main_domain, domain_a, domain_b, n,
                                 params, invert, device)


def _random_points_if_n_eq_1(main_domain, domain_a, domain_b, params, invert, device):
    final_points = torch.zeros((len(params), main_domain.dim), device=device)
    found_valid = torch.zeros((len(params), 1), dtype=bool, device=device) 
    while not all(found_valid):
        new_points = domain_a.sample_random_uniform(n=1, params=params, device=device)
        index_valid = _check_in_b(domain_b, params, invert, new_points)
        found_valid[index_valid] = True
        final_points[index_valid] = new_points.as_tensor[index_valid]
    return Points(final_points, main_domain.space)


def _random_points_inside(main_domain, domain_a, domain_b, n, params, invert, device):
    num_of_params = max(len(params), 1)
    warnings.warn(f"""Will sample random points in the created domain operation, with
                     a for loop over all input parameters, in total: {num_of_params}
                     This may slow down the training.""")
    random_points = Points.empty()
    for i in range(num_of_params):
        ith_params = params[i, ] if len(params) > 0 else Points.empty()
        number_valid = 0
        scaled_n = n
        while number_valid < n:
            # first create in a
            new_points = domain_a.sample_random_uniform(n=int(scaled_n),
                                                        params=ith_params, 
                                                        device=device)
            # check how many are in correct
            _, repeat_params = main_domain._repeat_params(len(new_points), ith_params)
            index_valid = _check_in_b(domain_b, repeat_params, invert, new_points)
            number_valid = len(index_valid)
            #scale up the number of point and try again
            scaled_n = 5*scaled_n if number_valid == 0 else scaled_n**2/number_valid + 1
        random_points = random_points | new_points[index_valid[:n], ]
    return random_points


def _inside_grid_with_n(main_domain, domain_a, domain_b, n, params, invert, device):
    """Creates a point grid inside of a cut or intersection domain.

    Parameters
    ----------
    main_domain : Domain
        The domain that represents the new created domain.
    domain_a, domain_b : Domain
        The two domains that define the main domain.
    n : int
        The number of points.
    params : Points
        Additional parameters for the domains.
    invert : bool
        Says if the points should lay in the domain_b (intersection) or if
        not (cut). For the Cut-Domain it is invert=True.
    device : str
        The device on which the points should be created.
    """
    # first sample grid inside the domain_a
    grid_a = domain_a.sample_grid(n=n, params=params, device=device)
    _, repeat_params = main_domain._repeat_params(n, params)
    index_valid = _check_in_b(domain_b, repeat_params, invert, grid_a)
    number_inside = len(index_valid)
    if number_inside == n:
        return grid_a
    # if the grid does not fit, scale the number of points
    scaled_n = int(n**2 / number_inside)
    grid_a = domain_a.sample_grid(n=scaled_n, params=params, device=device)
    _, repeat_params = main_domain._repeat_params(scaled_n, params)
    index_valid = _check_in_b(domain_b, repeat_params, invert, grid_a)
    grid_a = grid_a[index_valid, ]
    if len(grid_a) >= n:
        return grid_a[:n, ] 
    # add some random ones if still some missing
    rand_points = _random_points_inside(main_domain, domain_a, domain_b,
                                        n-len(grid_a), params, invert, device)
    return grid_a | rand_points


def _check_in_b(domain_b, params, invert, grid_a):
    #check what points are correct
    inside_b = domain_b._contains(grid_a, params)
    if invert:
      inside_b = torch.logical_not(inside_b)
    index = torch.where(inside_b)[0]
    return index


def _boundary_random_with_n(main_domain, domain_a, domain_b, n, params, device):
    """Creates a point grid on the boundary of a domain operation.

    Parameters
    ----------
    main_domain : Domain
        The domain that represents the new created domain.
    domain_a, domain_b : Domain
        The two domains that define the main domain.
    n : int
        The number of points.
    params : Points
        Additional parameters for the domains.
    device : str
        The device on which the points should be created.
    """
    if n == 1: 
        return _random_boundary_points_if_n_eq_1(main_domain, domain_a, domain_b,
                                                 params, device)
    return _random_points_boundary(main_domain, domain_a, domain_b, n, params, device)


def _random_boundary_points_if_n_eq_1(main_domain, domain_a, domain_b, params, device):
    final_points = torch.zeros((len(params), main_domain.dim+1), device=device)
    found_valid = torch.zeros((len(params), 1), dtype=bool, device=device) 
    boundaries = [domain_a.boundary, domain_b.boundary]
    use_b = False
    while not all(found_valid):
        new_points = \
            boundaries[use_b].sample_random_uniform(n=1, params=params, device=device)
        index_valid = main_domain._contains(new_points, params)
        index_valid = torch.logical_and(index_valid, torch.logical_not(found_valid))
        index_valid = torch.where(index_valid)[0]
        found_valid[index_valid] = True 
        final_points[index_valid] = new_points.as_tensor[index_valid]
        use_b = not use_b
    return Points(final_points, main_domain.space)


def _random_points_boundary(main_domain, domain_a, domain_b, n, params, device):
    num_of_params = max(len(params), 1)
    warnings.warn(f"""Will sample random points in the created domain operation, with
                     a for loop over all input parameters, in total: {num_of_params}
                     This may slow down the training.""")
    random_points = Points.empty()
    domains = [domain_a, domain_b]
    for i in range(num_of_params):
        ith_params = params[i, ] if len(params) > 0 else Points.empty()
        ith_points = Points.empty()
        # scale n such that the number of points corresponds to the size 
        # of the boundary
        sclaed_n = _compute_boundary_ratio(main_domain, domain_a,
                                           domain_b, ith_params, n, device=device)
        use_b = False # to switch between sampling on a and b
        while len(ith_points) < n:
            new_points = \
                domains[use_b].boundary.sample_random_uniform(n=sclaed_n[use_b],
                                                              params=ith_params, 
                                                              device=device)
            _, repeat_params = main_domain._repeat_params(len(new_points), ith_params)
            index_valid = torch.where(main_domain._contains(new_points, repeat_params))
            ith_points = ith_points | new_points[index_valid[0], ]
            use_b = not use_b # switch to other domain
        random_points = random_points | ith_points[:n, ]
    return random_points


def _compute_boundary_ratio(main_domain, domain_a, domain_b, ith_params, n, device='cpu'):
    main_volume = main_domain.volume(params=ith_params, device=device)
    a_volume = domain_a.boundary.volume(params=ith_params, device=device)
    b_volume = domain_b.boundary.volume(params=ith_params, device=device)
    return [int(n * a_volume/main_volume)+1, int(n * b_volume/main_volume)+1]


def _boundary_grid_with_n(main_domain, domain_a, domain_b, n, params, device):
    """Creates a point grid on the boundary of a domain operation.

    Parameters
    ----------
    main_domain : Domain
        The domain that represents the new created domain.
    domain_a, domain_b : Domain
        The two domains that define the main domain.
    n : int
        The number of points.
    params : Points
        Additional parameters for the domains.
    device : str
        The device on which the points should be created.
    """
    # first sample a grid on both boundaries
    grid_a = domain_a.boundary.sample_grid(n=n, params=params, device=device)
    grid_b = domain_b.boundary.sample_grid(n=n, params=params, device=device)  
    # check how many points are on the boundary of the operation domain
    on_bound_a, on_bound_b, a_correct, b_correct = \
        _check_points_on_main_boundary(main_domain, grid_a, grid_b, params)
    sum_of_correct = a_correct + b_correct
    if sum_of_correct == n:
        return grid_a[on_bound_a, ] | grid_b[on_bound_b, ]
    # scale the n so that more or fewer points are sampled and try again
    # to get a better grid. For the scaling we approximate the volume of the 
    # the main domain.
    a_surface = domain_a.boundary.volume(params, device=device)
    b_surface = domain_b.boundary.volume(params, device=device)
    approx_surface = a_surface * a_correct / n + b_surface * b_correct / n
    scaled_a = int(n * a_surface / approx_surface) + 1 # round up  
    scaled_b = max(int(n * b_surface / approx_surface), 1) # round to floor, but not 0
    grid_a = domain_a.boundary.sample_grid(n=scaled_a, params=params, device=device)
    grid_b = domain_b.boundary.sample_grid(n=scaled_b, params=params, device=device)  
    # check again how what points are correct and now just stay with this grid
    # if still some points are missing add random ones.
    on_bound_a, on_bound_b, a_correct, b_correct = \
        _check_points_on_main_boundary(main_domain, grid_a, grid_b, params)
    final_grid = grid_a[on_bound_a, ] | grid_b[on_bound_b, ]
    if len(final_grid) >= n:
        return final_grid[:n, ]
    rand_points = _random_points_boundary(main_domain, domain_a, domain_b,
                                          n-len(final_grid), params, device)
    return final_grid | rand_points


def _check_points_on_main_boundary(main_domain, grid_a, grid_b, params):
    _, repeat_params = main_domain._repeat_params(len(grid_a), params)
    on_bound_a = torch.where(main_domain._contains(grid_a, params=repeat_params))[0]
    _, repeat_params = main_domain._repeat_params(len(grid_b), params)
    on_bound_b = torch.where(main_domain._contains(grid_b, params=repeat_params))[0]
    a_correct = len(on_bound_a)
    b_correct = len(on_bound_b)
    return on_bound_a,on_bound_b, a_correct, b_correct