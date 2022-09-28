import torch

from ..domain import Domain, BoundaryDomain
from ...spaces import Points


class Interval(Domain):
    """Creates a Interval of the form [a, b].

    Parameters
    ----------
    space : Space
        The space in which this object lays.
    lower_bound : Number or callable
        The left/lower bound of the interval.
    upper_bound : Number or callable
        The right/upper bound of the interval.
    """
    def __init__(self, space, lower_bound, upper_bound):
        assert space.dim == 1
        lower_bound, upper_bound = self.transform_to_user_functions(lower_bound, upper_bound)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        super().__init__(space=space, dim=1)
        self.set_necessary_variables(self.lower_bound, self.upper_bound)

    def __call__(self, **data):
        new_lower_bound = self.lower_bound.partially_evaluate(**data)
        new_upper_bound = self.upper_bound.partially_evaluate(**data)
        return Interval(space=self.space, lower_bound=new_lower_bound, 
                        upper_bound=new_upper_bound)

    def _contains(self, points, params=Points.empty()):
        lb = self.lower_bound(points.join(params))
        ub = self.upper_bound(points.join(params))
        points = points[:, list(self.space.keys())].as_tensor
        bigger_then_low = torch.ge(points[:, None], lb) 
        smaller_then_up = torch.le(points[:, None], ub) 
        return torch.logical_and(bigger_then_low, smaller_then_up).reshape(-1, 1)

    def sample_random_uniform(self, n=None, d=None, params=Points.empty(),
                              device='cpu'):
        if d:
            n = self.compute_n_from_density(d, params)
        lb = self.lower_bound(params, device=device)
        ub = self.upper_bound(params, device=device)
        points = torch.rand((self.len_of_params(params), n, 1), device=device)
        points *= (ub - lb)
        points += lb
        return Points(points.reshape(-1, self.space.dim), self.space)

    def sample_grid(self, n=None, d=None, params=Points.empty(), device='cpu'):
        if d:
            n = self.compute_n_from_density(d, params)
        lb = self.lower_bound(params, device=device)
        ub = self.upper_bound(params, device=device)
        points = torch.linspace(0, 1, n+2, device=device)[1:-1, None]
        points = (ub - lb) * points 
        points += lb
        return Points(points.reshape(-1, self.space.dim), self.space)

    def bounding_box(self, params=Points.empty(), device='cpu'):
        lb = self.lower_bound(params, device=device)
        ub = self.upper_bound(params, device=device)
        return torch.stack((torch.min(lb), torch.max(ub)), dim=0)

    def _get_volume(self, params=Points.empty(), device='cpu'):
        lb = self.lower_bound(params, device=device)
        ub = self.upper_bound(params, device=device)
        return (ub - lb).reshape(-1, 1)

    @property
    def boundary(self):
        return IntervalBoundary(self)

    @property
    def boundary_left(self):
        """Returns only the left boundary value, useful for the definintion
        of inital conditions.
        """
        return IntervalSingleBoundaryPoint(self, side=self.lower_bound)

    @property
    def boundary_right(self):
        """Returns only the left boundary value, useful for the definintion
        of end conditions.
        """
        return IntervalSingleBoundaryPoint(self, side=self.upper_bound, normal_vec=1)


class IntervalBoundary(BoundaryDomain):

    def __init__(self, domain):
        assert isinstance(domain, Interval)
        super().__init__(domain)
    
    def _contains(self, points, params=Points.empty()):
        close_to_left, close_to_right = self._check_close_left_right(points, params) 
        return torch.logical_or(close_to_left, close_to_right).reshape(-1, 1)

    def _check_close_left_right(self, points, params):
        lb = self.domain.lower_bound(points.join(params))
        ub = self.domain.upper_bound(points.join(params))
        points = points[:, list(self.space.keys())].as_tensor
        close_to_left = torch.isclose(points[:, None], lb)
        close_to_right = torch.isclose(points[:, None], ub)
        return close_to_left, close_to_right

    def sample_random_uniform(self, n=None, d=None, params=Points.empty(),
                              device='cpu'):
        if d:
            n = self.compute_n_from_density(d, params)
        lb = self.domain.lower_bound(params, device=device)
        ub = self.domain.upper_bound(params, device=device)
        rand_side = torch.rand((self.len_of_params(params), n, 1), device=device)
        random_boundary_index = rand_side < 0.5 
        points = torch.where(random_boundary_index, lb, ub)
        return Points(points.reshape(-1, self.space.dim), self.space)

    def sample_grid(self, n=None, d=None, params=Points.empty(), device='cpu'):
        if d:
            n = self.compute_n_from_density(d, params)
        lb = self.domain.lower_bound(params, device)
        ub = self.domain.upper_bound(params, device)
        b_index = torch.tensor([0, 1], dtype=bool, device=device).repeat(int(n/2.0) + 1)
        points = torch.where(b_index[:n], lb, ub)
        return Points(points.reshape(-1, self.space.dim), self.space)

    def normal(self, points, params=Points.empty(), device='cpu'):
        points, params, device = \
            self._transform_input_for_normals(points, params, device)
        close_to_left, _ = self._check_close_left_right(points, params)
        return torch.where(close_to_left, -1, 1).reshape(-1, 1)

    def _get_volume(self, params=Points.empty(), device='cpu'):
        no_of_params = self.len_of_params(params)
        return 2 * torch.ones((no_of_params, 1), device=device)


class IntervalSingleBoundaryPoint(BoundaryDomain):

    def __init__(self, domain, side, normal_vec=-1):
        assert isinstance(domain, Interval)
        super().__init__(domain)
        self.side = side
        self.normal_vec = normal_vec

    def __call__(self, **data):
        evaluate_domain = self.domain(**data)
        return IntervalSingleBoundaryPoint(evaluate_domain, side=self.side,
                                           normal_vec=self.normal_vec)

    def _contains(self, points, params=Points.empty()):
        side = self.side(points.join(params))
        points = points[:, list(self.space.keys())].as_tensor
        inside = torch.isclose(points[:, None], side)
        return inside.reshape(-1, 1)

    def sample_random_uniform(self, n=None, d=None, params=Points.empty(),
                              device='cpu'):
        if d:
            n = self.compute_n_from_density(d, params)
        side = self.side(params, device=device)
        points = torch.ones((self.len_of_params(params), n, 1), device=device)
        points *= side
        return Points(points.reshape(-1, self.space.dim), self.space)

    def sample_grid(self, n=None, d=None, params=Points.empty(), device='cpu'):
        return self.sample_random_uniform(n=n, d=d, params=params, device=device)

    def normal(self, points, params=Points.empty(), device='cpu'):
        points, params, device = \
            self._transform_input_for_normals(points, params, device)
        points = torch.ones((self.len_of_params(points.join(params)), 1), device=device)
        return points * self.normal_vec

    def _get_volume(self, params=Points.empty(), device='cpu'):
        no_of_params = self.len_of_params(params)
        return 1 * torch.ones((no_of_params, 1), device=device)