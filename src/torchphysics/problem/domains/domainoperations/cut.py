import warnings
import torch

from ..domain import Domain, BoundaryDomain
from ...spaces import Points
from .sampler_helper import (_boundary_grid_with_n, _inside_grid_with_n, 
                             _inside_random_with_n, _boundary_random_with_n)


class CutDomain(Domain):
    """Implements the logical cut of two domains.

    Parameters
    ----------
    domain_a : Domain
        The first domain.
    domain_b : Domain
        The second domain. 
    """
    def __init__(self, domain_a: Domain, domain_b: Domain, contained=False):
        assert domain_a.space == domain_b.space
        self.domain_a = domain_a
        self.domain_b = domain_b
        self.contained = contained
        super().__init__(space=domain_a.space, dim=domain_a.dim)
        self.necessary_variables = domain_a.necessary_variables.copy()
        self.necessary_variables.update(domain_b.necessary_variables)

    def __call__(self, **data):
        domain_a = self.domain_a(**data)
        domain_b = self.domain_b(**data)
        return CutDomain(domain_a, domain_b)

    def _contains(self, points, params=Points.empty()):
        in_a = self.domain_a._contains(points, params)
        in_b = self.domain_b._contains(points, params)
        return torch.logical_and(in_a, torch.logical_not(in_b))

    def _get_volume(self, params=Points.empty(), device='cpu'):
        if not self.contained:
            warnings.warn("""Exact volume of this cut is not known, will use the
                             estimate: volume = domain_a.volume.
                             If you need the exact volume for sampling,
                             use domain.set_volume()""")
            return self.domain_a.volume(params, device=device)
        volume_a = self.domain_a.volume(params, device=device)
        volume_b = self.domain_b.volume(params, device=device)
        return volume_a - volume_b

    def bounding_box(self, params=Points.empty(), device='cpu'):
        return self.domain_a.bounding_box(params, device=device)

    def sample_random_uniform(self, n=None, d=None, params=Points.empty(), 
                              device='cpu'):
        if n:
            return _inside_random_with_n(self, self.domain_a, self.domain_b, n=n,
                                         params=params, invert=True, device=device)
        return self._sample_random_with_d(d, params, device)

    def _sample_random_with_d(self, d, params=Points.empty(), device='cpu'):
        points_a = self.domain_a.sample_random_uniform(d=d, params=params,
                                                       device=device)
        return self._cut_points(points_a, params)

    def _cut_points(self, points_a, params=Points.empty()):
        # check which points are in domain b
        n = len(points_a)
        _, repeated_params = self._repeat_params(n, params)
        in_b = self.domain_b._contains(points=points_a, params=repeated_params)    
        index = torch.where(torch.logical_not(in_b))[0]
        return points_a[index, ]

    def sample_grid(self, n=None, d=None, params=Points.empty(), device='cpu'):
        if n:
            return _inside_grid_with_n(self, self.domain_a, self.domain_b, n=n,
                                       params=params, invert=True, device=device)
        return self._sample_grid_with_d(d, params, device)

    def _sample_grid_with_d(self, d, params=Points.empty(), device='cpu'):
        points_a = self.domain_a.sample_grid(d=d, params=params, device=device)
        return self._cut_points(points_a, params)

    @property
    def boundary(self):
        return CutBoundaryDomain(self)


class CutBoundaryDomain(BoundaryDomain):

    def __init__(self, domain: CutDomain):
        assert not isinstance(domain.domain_a, BoundaryDomain)
        assert not isinstance(domain.domain_b, BoundaryDomain)
        super().__init__(domain)

    def _contains(self, points, params=Points.empty()):
        in_a = self.domain.domain_a._contains(points, params)
        in_b = self.domain.domain_b._contains(points, params)
        on_a_bound = self.domain.domain_a.boundary._contains(points, params)
        on_b_bound = self.domain.domain_b.boundary._contains(points, params)
        on_a_part = torch.logical_and(on_a_bound, torch.logical_not(in_b))
        on_b_part = torch.logical_and(on_b_bound, in_a)
        on_b_part = torch.logical_and(on_b_part, torch.logical_not(on_a_bound))
        return torch.logical_or(on_a_part, on_b_part)

    def _get_volume(self, params=Points.empty(), device='cpu'):
        if not self.domain.contained:
            warnings.warn("""Exact volume of this domain boundary is not known, 
                             will use the estimate: 
                             volume = domain_a.volume + domain_b.volume.
                             If you need the exact volume for sampling,
                             use domain.set_volume().""")
        volume_a = self.domain.domain_a.boundary.volume(params, device=device)
        volume_b = self.domain.domain_b.boundary.volume(params, device=device)
        return volume_a + volume_b
    
    def sample_random_uniform(self, n=None, d=None, params=Points.empty(), 
                              device='cpu'):
        if n:
            return _boundary_random_with_n(self, self.domain.domain_a, 
                                           self.domain.domain_b, n,
                                           params, device=device)
        return self._sample_random_with_d(d, params, device)

    def _sample_random_with_d(self, d, params=Points.empty(), device='cpu'):
        points_a = self.domain.domain_a.boundary.sample_random_uniform(d=d,
                                                                       device=device,
                                                                       params=params)
        points_a = self.domain._cut_points(points_a, params)
        points_b = self.domain.domain_b.boundary.sample_random_uniform(d=d,
                                                                       params=params, 
                                                                       device=device)  
        points_b = self._delete_outer_points(points_b, self.domain.domain_a, params)     
        return points_a | points_b

    def _delete_outer_points(self, points, domain, params=Points.empty()):
        n = len(points)
        _, repeated_params = self._repeat_params(n, params)
        inside = domain._contains(points, repeated_params)
        on_bound = domain.boundary._contains(points, repeated_params)
        inside = torch.logical_and(inside, torch.logical_not(on_bound))
        index = torch.where(inside)[0]
        return points[index, ]

    def sample_grid(self, n=None, d=None, params=Points.empty(), device='cpu'):
        if n:
            return _boundary_grid_with_n(self, self.domain.domain_a, 
                                         self.domain.domain_b, n, params, device)
        return self._sample_grid_with_d(d, params, device)

    def _sample_grid_with_d(self, d, params=Points.empty(), device='cpu'):
        points_a = self.domain.domain_a.boundary.sample_grid(d=d, params=params, 
                                                             device=device)
        points_a = self.domain._cut_points(points_a, params)
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
        normals = torch.where(on_a, a_normals, -b_normals)
        return normals