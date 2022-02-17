import warnings
import torch

from ..domain import Domain, BoundaryDomain
from ...spaces import Points
from .sampler_helper import (_boundary_grid_with_n, _inside_grid_with_n, 
                             _inside_random_with_n, _boundary_random_with_n)


class IntersectionDomain(Domain):
    """Implements the logical intersection of two domains.

    Parameters
    ----------
    domain_a : Domain
        The first domain.
    domain_b : Domain
        The second domain. 
    """
    def __init__(self, domain_a: Domain, domain_b: Domain):
        assert domain_a.space == domain_b.space
        self.domain_a = domain_a
        self.domain_b = domain_b
        super().__init__(space=domain_a.space, dim=domain_a.dim)
        self.necessary_variables = domain_a.necessary_variables.copy()
        self.necessary_variables.update(domain_b.necessary_variables)

    def __call__(self, **data):
        domain_a = self.domain_a(**data)
        domain_b = self.domain_b(**data)
        return IntersectionDomain(domain_a, domain_b)

    def _contains(self, points, params=Points.empty()):
        in_a = self.domain_a._contains(points, params)
        in_b = self.domain_b._contains(points, params)
        return torch.logical_and(in_a, in_b)

    def _get_volume(self, params=Points.empty(), device='cpu'):
        warnings.warn("""Exact volume of this intersection is not known,
                         will use the estimate: volume = domain_a.volume.
                         If you need the exact volume for sampling,
                         use domain.set_volume()""")
        return self.domain_a.volume(params, device=device)

    def bounding_box(self, params=Points.empty(), device='cpu'):
        bounds_a = self.domain_a.bounding_box(params, device=device)
        bounds_b = self.domain_b.bounding_box(params, device=device)
        bounds = []
        for i in range(self.space.dim):
            bounds.append(max([bounds_a[2*i], bounds_b[2*i]]))
            bounds.append(min([bounds_a[2*i+1], bounds_b[2*i+1]]))
        return torch.tensor(bounds, device=device)

    def sample_random_uniform(self, n=None, d=None, params=Points.empty(), 
                              device='cpu'):
        if n:
            return _inside_random_with_n(self, self.domain_a, self.domain_b, 
                                         n=n, params=params, invert=False,
                                         device=device)
        return self._sample_random_with_d(d, params, device)

    def _sample_random_with_d(self, d, params=Points.empty(), device='cpu'):
        points_a = self.domain_a.sample_random_uniform(d=d, params=params, device=device)
        return self._cut_points(points_a, params)

    def _cut_points(self, points, params=Points.empty()):
        # check which points are in domain b
        n = len(params)
        _, repeated_params = self._repeat_params(n, params)
        in_b = self.domain_b._contains(points=points, params=repeated_params)    
        index = torch.where(in_b)[0]
        return points[index, ]

    def sample_grid(self, n=None, d=None, params=Points.empty(), device='cpu'):
        if n:
            return _inside_grid_with_n(self, self.domain_a, self.domain_b, 
                                       n=n, params=params, invert=False, 
                                       device=device)
        return self._sample_grid_with_d(d, params, device)

    def _sample_grid_with_d(self, d, params=Points.empty(), device='cpu'):
        points_a = self.domain_a.sample_grid(d=d, params=params, device=device)
        return  self._cut_points(points_a, params)

    @property
    def boundary(self):
        return IntersectionBoundaryDomain(self)


class IntersectionBoundaryDomain(BoundaryDomain):

    def __init__(self, domain: IntersectionDomain):
        assert not isinstance(domain.domain_a, BoundaryDomain)
        assert not isinstance(domain.domain_b, BoundaryDomain)
        super().__init__(domain)

    def _contains(self, points, params=Points.empty()):
        in_a = self.domain.domain_a._contains(points, params)
        in_b = self.domain.domain_b._contains(points, params)
        on_a_bound = self.domain.domain_a.boundary._contains(points, params)
        on_b_bound = self.domain.domain_b.boundary._contains(points, params)
        on_a_part = torch.logical_and(on_a_bound, in_b)
        on_b_part = torch.logical_and(on_b_bound, in_a)
        return torch.logical_or(on_a_part, on_b_part)

    def _get_volume(self, params=Points.empty(), device='cpu'):
        warnings.warn("""Exact volume of this intersection-boundary is not known,
                         will use the estimate: volume = boundary_a + bounadry_b.
                         If you need the exact volume for sampling,
                         use domain.set_volume()""")
        volume_a = self.domain.domain_a.boundary.volume(params, device=device)
        volume_b = self.domain.domain_b.boundary.volume(params, device=device)
        return volume_a + volume_b

    def sample_random_uniform(self, n=None, d=None, params=Points.empty(),
                              device='cpu'):
        if n:
            return _boundary_random_with_n(self, self.domain.domain_a, 
                                           self.domain.domain_b, n, params, 
                                           device=device)
        return self._sample_random_with_d(d, params, device)


    def _sample_random_with_d(self, d, params=Points.empty(), device='cpu'):
        points_a = self.domain.domain_a.boundary.sample_random_uniform(d=d,
                                                                       params=params, 
                                                                       device=device)
        points_a = self._delete_outer_points(points_a, self.domain.domain_b, params) 
        points_b = self.domain.domain_b.boundary.sample_random_uniform(d=d,
                                                                       params=params, 
                                                                       device=device)  
        points_b = self._delete_outer_points(points_b, self.domain.domain_a, params)     
        return points_a | points_b 

    def _delete_outer_points(self, points, domain, params):
        n = len(points)
        _, repeated_params = self._repeat_params(n, params)
        inside = domain._contains(points, repeated_params)
        index = torch.where(inside)[0]
        return points[index, ]

    def sample_grid(self, n=None, d=None, params=Points.empty(), device='cpu'):
        if n:
            return _boundary_grid_with_n(self, self.domain.domain_a, 
                                         self.domain.domain_b, n, params, 
                                         device=device)
        return self._sample_grid_with_d(d, params, device)

    def _sample_grid_with_d(self, d, params=Points.empty(), device='cpu'):
        points_a = self.domain.domain_a.boundary.sample_grid(d=d, params=params, 
                                                             device=device)
        points_a = self._delete_outer_points(points_a, self.domain.domain_b, params) 
        points_b = self.domain.domain_b.boundary.sample_grid(d=d, params=params, 
                                                             device=device)  
        points_b = self._delete_outer_points(points_b, self.domain.domain_a, params)     
        return points_a | points_b    

    def normal(self, points, params=Points.empty(), device='cpu'):
        points, params, device = \
            self._transform_input_for_normals(points, params, device)
        a_normals = self.domain.domain_a.boundary.normal(points, params, device)
        b_normals = self.domain.domain_b.boundary.normal(points, params, device)
        on_a = self.domain.domain_a.boundary._contains(points, params)
        normals = torch.where(on_a, a_normals, b_normals)
        return normals