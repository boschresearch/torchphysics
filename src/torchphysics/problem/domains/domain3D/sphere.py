import torch
import numpy as np

from ..domain import Domain, BoundaryDomain
from ...spaces import Points


class Sphere(Domain):
    """Class for a sphere.

    Parameters
    ----------
    space : Space
        The space in which this object lays.
    center : array_like or callable
        The center of the sphere, e.g. center = [5, 0, 0].
    radius : number or callable
        The radius of the sphere.
    """
    def __init__(self, space, center, radius):
        assert space.dim ==  3
        center, radius = self.transform_to_user_functions(center, radius)
        self.center = center
        self.radius = radius
        super().__init__(space=space, dim=3)
        self.set_necessary_variables(self.radius, self.center)

    def __call__(self, **data):
        new_center = self.center.partially_evaluate(**data)
        new_radius = self.radius.partially_evaluate(**data)
        return Sphere(space=self.space, center=new_center, radius=new_radius)

    def _compute_center_and_radius(self, params=Points.empty(), device='cpu'):
        center = self.center(params, device).reshape(-1, 3)
        radius = self.radius(params, device)
        return center,radius

    def _contains(self, points, params=Points.empty()):
        center, radius = self._compute_center_and_radius(points.join(params), device=points.device)
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

    def _get_volume(self, params=Points.empty(), device='cpu'):
        radius = self.radius(params, device=device)
        volume = 3.0/4.0 * np.pi * radius**3
        return volume.reshape(-1, 1)

    def sample_random_uniform(self, n=None, d=None, params=Points.empty(),
                              device='cpu'):
        if d:
            n = self.compute_n_from_density(d, params)
        center, radius = self._compute_center_and_radius(params, device)
        num_of_params = self.len_of_params(params)
        # take cubic root to stay uniform
        r = torch.pow(torch.rand((num_of_params, n, 1), device=device), 1/3.0)
        r *= radius
        phi = 2 * np.pi * torch.rand((num_of_params, n, 1), device=device)
        theta = torch.rand((num_of_params, n, 1), device=device)
        theta = torch.arccos(2*theta - 1) - np.pi/2.0
        x = torch.multiply(torch.multiply(r, torch.cos(phi)), torch.cos(theta))
        y = torch.multiply(torch.multiply(r, torch.sin(phi)), torch.cos(theta))
        z = torch.multiply(r, torch.sin(theta))
        points = torch.cat((x, y, z), dim=2)
        # [:,None,:] is needed so that the correct entries will be added
        points += center[:, None, :]
        return Points(points.reshape(-1, self.space.dim), self.space)

    def sample_grid(self, n=None, d=None, params=Points.empty(), device='cpu'):
        if d:
            n = self.compute_n_from_density(d, params)
        if n > 10:
            # for to small n the sampling in the box is not stable,
            # in this case only use random points.
            center, radius = self._compute_center_and_radius(params, device)
            points = self._point_grid_in_box(n, radius.item(), device)
            points_inside = self._get_points_inside(points, radius.item())
            points_inside += center
        else:
            points_inside = torch.empty((0, self.dim), device=device)
        finals_points = self._append_random(points_inside, n, params, device)
        return Points(finals_points, self.space)

    def _point_grid_in_box(self, n, radius, device):
        scaled_n = int(np.ceil(np.cbrt(n*6/np.pi)))
        axis = torch.linspace(-radius, radius, scaled_n, device=device)
        points = torch.stack(torch.meshgrid(axis, axis, axis)).T
        return points.reshape(-1, 3)

    def _get_points_inside(self, points, radius):
        norm = torch.linalg.norm(points, dim=1).reshape(-1, 1)
        inside = (norm <= radius)
        index = torch.where(inside)[0]
        return points[index]

    def _append_random(self, points_inside, n, params, device):
        if len(points_inside) == n:
            return points_inside
        random_points = self.sample_random_uniform(n=n-len(points_inside),
                                                   params=params, device=device)
        random_points = random_points[:, list(self.space.keys())].as_tensor
        return torch.cat((points_inside, random_points), dim=0)

    @property
    def boundary(self):
        return SphereBoundary(self)


class SphereBoundary(BoundaryDomain):

    def __init__(self, domain):
        assert isinstance(domain, Sphere)
        super().__init__(domain)

    def _contains(self, points, params=Points.empty()):
        center, radius = self.domain._compute_center_and_radius(points.join(params), device=points.device)
        points = points[:, list(self.space.keys())].as_tensor
        norm = torch.linalg.norm(points - center, dim=1).reshape(-1, 1)
        return torch.isclose(norm[:, None], radius).reshape(-1, 1)

    def _get_volume(self, params=Points.empty(), device='cpu'):
        radius = self.domain.radius(params, device=device)
        volume = 4 * np.pi * radius**2
        return volume.reshape(-1, 1)

    def sample_random_uniform(self, n=None, d=None, params=Points.empty(),
                              device='cpu'):
        if d:
            n = self.compute_n_from_density(d, params)
        center, radius = self.domain._compute_center_and_radius(params, device)
        num_of_params = self.len_of_params(params)
        phi = 2 * np.pi * torch.rand((num_of_params, n, 1), device=device)
        theta = torch.rand((num_of_params, n, 1), device=device)
        theta = torch.arccos(2*theta - 1) - np.pi/2.0
        x = torch.multiply(torch.multiply(radius, torch.cos(phi)), torch.cos(theta))
        y = torch.multiply(torch.multiply(radius, torch.sin(phi)), torch.cos(theta))
        z = torch.multiply(radius, torch.sin(theta))
        points = torch.cat((x, y, z), dim=2)
        points += center[:, None, :]
        return Points(points.reshape(-1, self.space.dim), self.space)

    def sample_grid(self, n=None, d=None, params=Points.empty(), device='cpu'):
        if d:
            n = self.compute_n_from_density(d, params)
        center, radius = self.domain._compute_center_and_radius(params, device)
        # From: https://stackoverflow.com/questions/9600801/
        # evenly-distributing-n-points-on-a-sphere
        points = []
        # Use Fibonacci-Sphere for radius = 1, and then scale this sphere
        phi = np.pi * (3.0 - np.sqrt(5.0)) # golden angle in radians
        index = torch.arange(0, n, device=device)
        y = 1 - index / (n-1) * 2  # y goes from 1 to -1
        current_radius = torch.sqrt(1 - y**2)
        theta = phi * index
        x = current_radius * torch.cos(theta)
        z = current_radius * torch.sin(theta)
        points = torch.column_stack((x, y, z)).reshape(-1, 3)
        # Translate to center and scale
        points *= radius.item()
        points = torch.add(points, center)
        return Points(points, self.space)

    def normal(self, points, params=Points.empty(), device='cpu'):
        points, params, device = \
            self._transform_input_for_normals(points, params, device)
        center, radius = self.domain._compute_center_and_radius(points.join(params), device)
        points = points[:, list(self.space.keys())].as_tensor
        normal = points - center
        return torch.divide(normal[:, None], radius).reshape(-1, 3)