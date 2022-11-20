import torch
import numpy as np

from ..domain import Domain, BoundaryDomain
from ...spaces import Points


class Circle(Domain):
    """Class for circles.

    Parameters
    ----------
    space : Space
        The space in which this object lays.
    center : array_like or callable
        The center of the circle, e.g. center = [5,0].
    radius : number or callable
        The radius of the circle.
    """   
    def __init__(self, space, center, radius):
        assert space.dim == 2
        center, radius = self.transform_to_user_functions(center, radius)
        self.center = center
        self.radius = radius
        super().__init__(space=space, dim=2)
        self.set_necessary_variables(self.radius, self.center)

    def __call__(self, **data):
        new_center = self.center.partially_evaluate(**data)
        new_radius = self.radius.partially_evaluate(**data)
        return Circle(space=self.space, center=new_center, radius=new_radius)

    def _contains(self, points, params=Points.empty()):
        center, radius = self._compute_center_and_radius(points.join(params), points.device)
        points = points[:, list(self.space.keys())].as_tensor
        norm = torch.linalg.norm(points - center, dim=1).reshape(-1, 1)
        return torch.le(norm[:, None], radius).reshape(-1, 1)

    def bounding_box(self, params=Points.empty(), device='cpu'):
        center, radius = self._compute_center_and_radius(params, device=device)
        bounds = []
        for i in range(self.dim):
            i_min = torch.min(center[:, i] - radius)
            i_max = torch.max(center[:, i] + radius)
            bounds.append(i_min.item())
            bounds.append(i_max.item())
        return torch.tensor(bounds, device=device)

    def sample_random_uniform(self, n=None, d=None, params=Points.empty(),
                              device='cpu'):
        if d:
            n = self.compute_n_from_density(d, params)
        center, radius = self._compute_center_and_radius(params, device=device)
        num_of_params = self.len_of_params(params)
        r = torch.sqrt(torch.rand((num_of_params, n, 1), device=device))
        r *= radius
        phi = 2 * np.pi * torch.rand((num_of_params, n, 1), device=device)
        points = torch.cat((torch.multiply(r, torch.cos(phi)),
                            torch.multiply(r, torch.sin(phi))), dim=2)
        # [:,None,:] is needed so that the correct entries will be added
        points += center[:, None, :]
        return Points(points.reshape(-1, self.space.dim), self.space)

    def sample_grid(self, n=None, d=None, params=Points.empty(), device='cpu'):
        if d:
            n = self.compute_n_from_density(d, params)
        center, radius = self._compute_center_and_radius(params, device)
        num_of_params = self.len_of_params(params)
        grid = self._equidistant_points_in_circle(n, device=device)
        grid = grid.repeat(num_of_params, 1).view(num_of_params, n, 2) 
        points = torch.multiply(radius, grid)
        points += center[:, None, :]
        return Points(points.reshape(-1, self.space.dim), self.space)

    def _compute_center_and_radius(self, params=Points.empty(), device='cpu'):
        center = self.center(params, device=device).reshape(-1, 2)
        radius = self.radius(params, device=device)
        return center,radius

    def _equidistant_points_in_circle(self, n, device):
        # use a sunflower seed arrangement:
        # https://demonstrations.wolfram.com/SunflowerSeedArrangements/
        gr = (np.sqrt(5) + 1)/2.0 # golden ratio
        points = torch.arange(1, n+1, device=device)
        phi = (2 * np.pi / gr) * points
        radius = torch.sqrt(points - 0.5) / np.sqrt(n + 0.5) 
        points = torch.column_stack((torch.multiply(radius, torch.cos(phi)),
                                     torch.multiply(radius, torch.sin(phi))))
        return points                             

    def _get_volume(self, params=Points.empty(), device='cpu'):
        radius = self.radius(params, device=device)
        volume = np.pi * radius**2
        return volume.reshape(-1, 1)

    @property
    def boundary(self):
        return CircleBoundary(self)


class CircleBoundary(BoundaryDomain):

    def __init__(self, domain):
        assert isinstance(domain, Circle)
        super().__init__(domain)

    def _contains(self, points, params=Points.empty()):
        center, radius = self.domain._compute_center_and_radius(points.join(params), points.device)
        points = points[:, list(self.space.keys())].as_tensor
        norm = torch.linalg.norm(points - center, dim=1).reshape(-1, 1)
        return torch.isclose(norm[:, None], radius).reshape(-1, 1)

    def sample_random_uniform(self, n=None, d=None, params=Points.empty(), 
                              device='cpu'):
        if d:
            n = self.compute_n_from_density(d, params)
        center, radius = self.domain._compute_center_and_radius(params, device)
        phi = 2 * np.pi * torch.rand((self.len_of_params(params), n, 1), device=device)
        points = torch.cat((torch.multiply(radius, torch.cos(phi)),
                            torch.multiply(radius, torch.sin(phi))), 
                            dim=2)
        points += center[:, None, :]
        return Points(points.reshape(-1, self.space.dim), self.space)

    def sample_grid(self, n=None, d=None, params=Points.empty(), device='cpu'):
        if d:
            n = self.compute_n_from_density(d, params)
        center, radius = self.domain._compute_center_and_radius(params, device)
        num_of_params = self.len_of_params(params)
        grid = torch.linspace(0, 2*np.pi, n+1, device=device)[:-1] # last one would be double
        phi = grid.repeat(num_of_params).view(num_of_params, n, 1) 
        points = torch.cat((torch.multiply(radius, torch.cos(phi)),
                            torch.multiply(radius, torch.sin(phi))), 
                            dim=2)
        points += center[:, None, :]
        return Points(points.reshape(-1, self.space.dim), self.space)

    def normal(self, points, params=Points.empty(), device='cpu'):
        points, params, device = \
            self._transform_input_for_normals(points, params, device)
        center, radius = self.domain._compute_center_and_radius(points.join(params), device)
        points = points[:, list(self.space.keys())].as_tensor
        normal = (points - center)
        return torch.divide(normal[:, None], radius).reshape(-1, 2)

    def _get_volume(self, params=Points.empty(), device='cpu'):
        radius = self.domain.radius(params, device=device)
        volume = 2 * np.pi * radius
        return volume.reshape(-1, 1)
