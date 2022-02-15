import warnings
import torch

from ..domain import Domain, BoundaryDomain
from ...spaces import Points
from .sampler_helper import _boundary_grid_with_n, _boundary_random_with_n


class UnionDomain(Domain):
    """Implements the logical union of two domains.

    Parameters
    ----------
    domain_a : Domain
        The first domain.
    domain_b : Domain
        The second domain. 
    """    
    def __init__(self, domain_a: Domain, domain_b: Domain, disjoint=False):
        assert domain_a.space == domain_b.space
        self.domain_a = domain_a
        self.domain_b = domain_b
        self.disjoint = disjoint
        super().__init__(space=domain_a.space, dim=domain_a.dim)
        self.necessary_variables = domain_a.necessary_variables.copy()
        self.necessary_variables.update(domain_b.necessary_variables)

    def _get_volume(self, params=Points.empty(), return_value_of_a_b=False, device='cpu'):
        if not self.disjoint:
            warnings.warn("""Exact volume of this union is not known, will use the
                             estimate: volume = domain_a.volume + domain_b.volume.
                             If you need the exact volume for sampling,
                             use domain.set_volume()""")
        volume_a = self.domain_a.volume(params, device=device)
        volume_b = self.domain_b.volume(params, device=device)
        if return_value_of_a_b:
            return volume_a + volume_b, volume_a, volume_b
        return volume_a + volume_b

    def _contains(self, points, params=Points.empty()):
        in_a = self.domain_a._contains(points, params)
        in_b = self.domain_b._contains(points, params)
        return torch.logical_or(in_a, in_b)

    def __call__(self, **data):
        domain_a = self.domain_a(**data)
        domain_b = self.domain_b(**data)
        return UnionDomain(domain_a, domain_b)

    def bounding_box(self, params=Points.empty(), device='cpu'):
        bounds_a = self.domain_a.bounding_box(params, device=device)
        bounds_b = self.domain_b.bounding_box(params, device=device)
        bounds = []
        for i in range(self.space.dim):
            bounds.append(min([bounds_a[2*i], bounds_b[2*i]]))
            bounds.append(max([bounds_a[2*i+1], bounds_b[2*i+1]]))
        return torch.tensor(bounds, device=device)

    def sample_random_uniform(self, n=None, d=None, params=Points.empty(), 
                              device='cpu'):
        if n:
            return self._sample_random_with_n(n, params, device)
        # esle d not None
        return self._sample_random_with_d(d, params, device)

    def _sample_random_with_n(self, n, params=Points.empty(), device='cpu'):
        # sample n points in both domains
        points_a = self.domain_a.sample_random_uniform(n=n, params=params, device=device)
        points_b = self.domain_b.sample_random_uniform(n=n, params=params, device=device)
        # check which points of domain b are in domain a
        _, repeated_params = self._repeat_params(n, params)
        in_a = self.domain_a._contains(points=points_b, params=repeated_params)
        # approximate volume of this domain
        volume_approx, volume_a, _ = self._get_volume(return_value_of_a_b=True,
                                                      params=repeated_params,
                                                      device=device)
        volume_ratio = torch.divide(volume_a, volume_approx)
        # choose points depending of the proportion of the domain w.r.t. the
        # whole domain union
        rand_index = torch.rand((max(n, len(repeated_params)), 1), device=device)
        rand_index = torch.logical_or(in_a, rand_index <= volume_ratio)
        points = torch.where(rand_index, points_a, points_b)                
        return Points(points, self.space)

    def _sample_random_with_d(self, d, params=Points.empty(), device='cpu'):
        # sample n points in both domains
        points_a = self.domain_a.sample_random_uniform(d=d, params=params, device=device)
        points_b = self.domain_b.sample_random_uniform(d=d, params=params, device=device)      
        return self._append_points(points_a, points_b, params)

    def _append_points(self, points_a, points_b, params=Points.empty()):
        in_a = self._points_lay_in_other_domain(points_b, self.domain_a, params)  
        # delete the points that are in domain a (so the sampling stays uniform)
        index = torch.where(torch.logical_not(in_a))[0]
        disjoint_b_points = points_b[index, ]    
        return points_a | disjoint_b_points

    def _points_lay_in_other_domain(self, points, domain, params=Points.empty()):
        # check which points of domain b are in domain a
        n = len(points)
        _, repeated_params = self._repeat_params(n, params)
        in_a = domain._contains(points=points, params=repeated_params)
        return in_a

    def sample_grid(self, n=None, d=None, params=Points.empty(), device='cpu'):
        if n:
            return self._sample_grid_with_n(n, params, device)
        # else d not None
        return self._sample_grid_with_d(d, params, device)

    def _sample_grid_with_n(self, n, params=Points.empty(), device='cpu'):
        volume_approx, volume_a, _ = self._get_volume(return_value_of_a_b=True,
                                                      params=params,
                                                      device=device)
        scaled_n = int(torch.ceil(n * volume_a/volume_approx))
        points_a = self.domain_a.sample_grid(n=scaled_n, params=params, 
                                             device=device)
        if n - scaled_n > 0:
            return self._sample_in_b(n, params, points_a, device)
        return points_a

    def _sample_in_b(self, n, params, points_a, device):
        # check how many points from domain a lay in b, these points will not be used!
        in_b = self._points_lay_in_other_domain(points_a, self.domain_b, params)
        index = torch.where(torch.logical_not(in_b))[0]
        scaled_n = n - len(index)
        points_b = self.domain_b.sample_grid(n=scaled_n, params=params, device=device)
        return points_a[index, ] | points_b

    def _sample_grid_with_d(self, d, params=Points.empty(), device='cpu'):
        points_a = self.domain_a.sample_grid(d=d, params=params, device=device)
        points_b = self.domain_b.sample_grid(d=d, params=params, device=device)      
        return self._append_points(points_a, points_b, params)

    @property
    def boundary(self):
        return UnionBoundaryDomain(self)


class UnionBoundaryDomain(BoundaryDomain):

    def __init__(self, domain: UnionDomain):
        assert not isinstance(domain.domain_a, BoundaryDomain)
        assert not isinstance(domain.domain_b, BoundaryDomain)
        super().__init__(domain)

    def _contains(self, points, params=Points.empty()):
        in_a = self.domain.domain_a._contains(points, params)
        in_b = self.domain.domain_b._contains(points, params)
        on_a_bound = self.domain.domain_a.boundary._contains(points, params)
        on_b_bound = self.domain.domain_b.boundary._contains(points, params)
        on_both = torch.logical_and(on_b_bound, on_a_bound)
        on_a_part = torch.logical_and(on_a_bound, torch.logical_not(in_b))
        on_b_part = torch.logical_and(on_b_bound, torch.logical_not(in_a))
        return torch.logical_or(on_a_part, torch.logical_or(on_b_part, on_both))

    def _get_volume(self, params=Points.empty(), device='cpu'):
        if not self.domain.disjoint:
            warnings.warn("""Exact volume of this domain is not known, will use the
                             estimate: volume = domain_a.volume + domain_b.volume.
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
                                           device)
        return self._sample_random_with_d(d, params, device)

    def _sample_random_with_d(self, d, params=Points.empty(), device='cpu'):
        points_a = self.domain.domain_a.boundary.sample_random_uniform(d=d,
                                                                       params=params, 
                                                                       device=device)
        points_a = self._delete_points_in_b(points_a, params)
        points_b = self.domain.domain_b.boundary.sample_random_uniform(d=d,
                                                                       params=params, 
                                                                       device=device)  
        points_b = self._delete_inner_points(points_b, self.domain.domain_a, params)     
        return points_a | points_b    

    def _delete_inner_points(self, points, domain, params=Points.empty()):
        _, repeated_params = self._repeat_params(len(points), params)
        inside = domain._contains(points, repeated_params)
        on_bound = domain.boundary._contains(points, repeated_params)
        valid_points = torch.logical_or(on_bound, torch.logical_not(inside))
        index = torch.where(valid_points)[0]
        return points[index, ]

    def _delete_points_in_b(self, points, params=Points.empty()):
        _, repeated_params = self._repeat_params(len(points), params)
        inside = self.domain.domain_b._contains(points, repeated_params)
        index = torch.where(torch.logical_not(inside))[0]
        return points[index, ]

    def sample_grid(self, n=None, d=None, params=Points.empty(), device='cpu'):
        if n:
            return _boundary_grid_with_n(self, self.domain.domain_a, 
                                         self.domain.domain_b, n, params, 
                                         device)
        return self._sample_grid_with_d(d, params, device)

    def _sample_grid_with_d(self, d, params=Points.empty(), device='cpu'):
        points_a = self.domain.domain_a.boundary.sample_grid(d=d, params=params, 
                                                             device=device)
        points_a = self._delete_points_in_b(points_a, params)
        points_b = self.domain.domain_b.boundary.sample_grid(d=d, params=params, 
                                                             device=device)  
        points_b = self._delete_inner_points(points_b, self.domain.domain_a, params)     
        return points_a | points_b    

    def normal(self, points, params=Points.empty(), device='cpu'):
        points, params, device = \
            self._transform_input_for_normals(points, params, device)
        a_normals = self.domain.domain_a.boundary.normal(points, params, device)
        b_normals = self.domain.domain_b.boundary.normal(points, params, device)
        on_a = self.domain.domain_a.boundary._contains(points, params)
        normals = torch.where(on_a, a_normals, b_normals)
        return normals