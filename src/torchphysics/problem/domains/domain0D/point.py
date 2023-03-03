import torch

from ..domain import Domain
from ...spaces import Points

class Point(Domain):
    """Creates a single point at the given coordinates.

    Parameters
    ----------
    space : Space
        The space in which this object lays.
    coord : Number, List or callable
        The coordinate of the point.
    """
    def __init__(self, space, point):
        self.bounding_box_tol = 0.1
        point = self.transform_to_user_functions(point)[0]
        self.point = point
        super().__init__(space=space, dim=0)
        self.set_necessary_variables(point)

    def __call__(self, **data):
        new_point = self.point.partially_evaluate(**data)
        return Point(space=self.space, point=new_point)

    def _contains(self, points, params=Points.empty()):
        point_params = self.point(points.join(params))
        points = points[:, list(self.space.keys())].as_tensor
        inside = torch.isclose(points[:, None], point_params, atol=0.001)
        return torch.all(inside, dim=2).reshape(-1, 1)

    def bounding_box(self, params=Points.empty(), device='cpu'):
        if callable(self.point.fun): # if point moves
             return self._bounds_for_callable_point(params, device=device)
        if isinstance(self.point.fun, (torch.Tensor, list)):
             return self._bounds_for_higher_dimensions(device=device)
        return torch.tensor([self.point.fun - self.bounding_box_tol, 
                self.point.fun + self.bounding_box_tol], device=device)

    def _bounds_for_callable_point(self, params, device='cpu'):
        bounds = []
        discrete__points = self.point(params, device=device).reshape(-1, self.space.dim)
        for i in range(self.space.dim):
            min_ = torch.min(discrete__points[:, i])
            max_ = torch.max(discrete__points[:, i])
            if min_ == max_:
                min_ -= self.bounding_box_tol
                max_ += self.bounding_box_tol
            bounds.append(min_.item()), bounds.append(max_.item())
        return torch.tensor(bounds, device=device)

    def _bounds_for_higher_dimensions(self, device='cpu'):
        bounds = []
        for i in range(self.space.dim):
            p = self.point.fun[i]
            # substract/add a value to get a real bounding box, 
            # important if we later use these values to normalize the input
            bounds.append(p - self.bounding_box_tol)
            bounds.append(p + self.bounding_box_tol)
        return torch.tensor(bounds, device=device)

    def sample_random_uniform(self, n=None, d=None, params=Points.empty(), device='cpu'):
        if d:
            n = self.compute_n_from_density(d, params)
        point_params = self.point(params, device=device)
        points = torch.ones((self.len_of_params(params), n, self.space.dim),
                            device=device)
        points *= point_params
        return Points(points.reshape(-1, self.space.dim), self.space)

    def sample_grid(self, n=None, d=None, params=Points.empty(), device='cpu'):
        # for one single point grid and random sampling is the same
        return self.sample_random_uniform(n=n, d=d, params=params, device=device)

    def _get_volume(self, params=Points.empty(), device='cpu'):
        no_of_params = self.len_of_params(params)
        return 1 * torch.ones((no_of_params, 1), device=device)