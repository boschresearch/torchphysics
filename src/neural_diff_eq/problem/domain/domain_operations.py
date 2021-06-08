import numpy as np
import abc
from .domain import Domain
from .domain1D import Interval


class Domain_operation(Domain):

    def __init__(self, dim, volume, surface, tol):
        super().__init__(dim, volume, surface, tol)

    def _check_correct_dim(self, base, compare):
        if base.dim != compare.dim:
            raise ValueError('Domains have to be of the same dimension,' \
                             ' found dimensions %d and %d' %(base.dim, 
                             compare.dim))

    def _check_if_input_is_interval(self, base):
        if isinstance(base, Interval):
            raise ValueError('It is more efficient to create a new interval!')

    def _check_boundary_ratio(self, domain, n=100, type='grid'):
        points = domain.sample_boundary(n, type)
        number = len(points[np.where(self.is_on_boundary(points))])
        return number/n

    def _check_volume_ratio(self, domain_1, domain_2, n=100, type='grid'):
        points = domain_1.sample_inside(n, type)
        number = len(points[np.where(domain_2.is_inside(points))])
        return number/n

    def _sample_new_points_inside(self, domain_1, domain_2, n, type):
        new_points = domain_1.sample_inside(n, type)
        inside_2 = domain_2.is_inside(new_points)
        index = np.where(inside_2)[0]
        new_points = np.delete(new_points, index, axis=0)
        return new_points

    def _sample_new_points_boundary(self, domain_1, n, type):
        new_points = domain_1.sample_boundary(n, type)
        on_bound = self.is_on_boundary(new_points)
        index = np.where(np.invert(on_bound))[0]
        new_points = np.delete(new_points, index, axis=0)
        return new_points
 
    @abc.abstractmethod
    def _approximate_volume(self, n):
        return

    @abc.abstractmethod
    def _approximate_surface(self, n):
        return

    def _random_sampling_boundary(self, domain_1, domain_2, n):
        points = np.empty((0,self.dim))
        domains = [domain_1, domain_2]
        scaled_n = [int(np.ceil(domain_1.surface*n/self.surface)),
                    int(np.ceil(domain_2.surface*n/self.surface))]
        current_domain_is_base = True
        while len(points) < n:
            current_domain = domains[current_domain_is_base]
            current_n = scaled_n[current_domain_is_base]
            new_points = self._sample_new_points_boundary(current_domain, current_n,
                                                          type='random')
            points = np.append(points, new_points, axis=0)
            current_domain_is_base = not current_domain_is_base
        if len(points) > n:
            points = self._cut_points(points, n)
        return points.astype(np.float32)

    def _grid_sampling_boundary(self, domain_1, domain_2, n):
        points = np.empty((0,self.dim))
        base_n = int(domain_1.surface*n/self.surface)
        new_points = self._sample_new_points_boundary(domain_1, base_n, type='grid')
        points = np.append(points, new_points, axis=0)
        cut_n = int(domain_2.surface*n/self.surface)
        new_points = self._sample_new_points_boundary(domain_2, cut_n, type='grid')
        points = np.append(points, new_points, axis=0)
        if len(points) < n:
            points = np.append(points, self._random_sampling_boundary(n-len(points)),
                               axis=0)
        if len(points) > n:
            points = self._cut_points(points, n)
        return points.astype(np.float32)


class Cut(Domain_operation):
    
    def __init__(self, base, cut, n=100):
        self._check_correct_dim(base, cut)
        self._check_if_input_is_interval(base)
        self.base = base
        self.cut = cut        
        volume = self._approximate_volume(n)
        surface = self._approximate_surface(n)
        super().__init__(dim=base.dim, tol=base.tol, volume=volume,
                         surface=surface)
    
    def _approximate_volume(self, n):
        volume_ratio = self._check_volume_ratio(self.cut, self.base, n)
        return self.base.volume-volume_ratio*self.cut.volume
        
    def _approximate_surface(self, n):
        bound_ratio_base = self._check_boundary_ratio(self.base, n)
        bound_ratio_cut = self._check_boundary_ratio(self.cut, n)
        return self.base.surface*bound_ratio_base + self.cut.surface*bound_ratio_cut 

    def is_inside(self, points):
        in_base = self.base.is_inside(points)
        in_cut = self.cut.is_inside(points)
        inside = np.logical_and(in_base, np.invert(in_cut))
        on_cut_bound = self.cut.is_on_boundary(points)
        on_cut_bound = np.logical_and(on_cut_bound, in_base)
        return np.logical_and(inside, np.invert(on_cut_bound)) 

    def is_on_boundary(self, points):
        on_base = self.base.is_on_boundary(points)
        on_cut = self.cut.is_on_boundary(points)
        in_base = self.base.is_inside(points)
        in_cut = self.cut.is_inside(points)
        on_base_bound = np.logical_and(on_base, np.invert(in_cut))
        on_cut_bound = np.logical_and(on_cut, in_base)
        on_cut_bound = np.logical_and(on_cut_bound, np.invert(on_base))
        return np.logical_or(on_base_bound, on_cut_bound)

    def boundary_normal(self, points):
        normals = np.zeros((len(points), self.dim))
        on_base = self.base.is_on_boundary(points)
        on_cut = self.cut.is_on_boundary(points)
        if any(on_base):
            index_base = np.where(on_base)[0]
            normals[index_base] += self.base.boundary_normal(points[index_base])
        if any(on_cut):
            index_cut = np.where(on_cut)[0]
            normals[index_cut] -= self.cut.boundary_normal(points[index_cut])
        index_both = np.where(np.logical_and(on_base, on_cut))[0]
        normals[index_both] *= 1/np.sqrt(2)  
        return normals.astype(np.float32)

    def grid_for_plots(self, n):
        base_n = int(np.ceil(self.base.volume/self.volume * n))
        points = self.base.grid_for_plots(base_n)
        inside_cut = self.cut.is_inside(points)
        index = np.where(np.invert(inside_cut))[0]
        return points[index]
  
    def _random_sampling_inside(self, n):
        points = np.empty((0,self.dim))
        while n > 0:
            new_points = self._sample_new_points_inside(self.base, self.cut,
                                                        n, type='random')
            points = np.append(points, new_points, axis=0)
            n -= len(new_points)
        return points.astype(np.float32)

    def _grid_sampling_inside(self, n):
        scaled_n = int(self.base.volume/self.volume * n)
        points = np.empty((0,self.dim))
        new_points = self._sample_new_points_inside(self.base, self.cut,
                                                    scaled_n, type='grid')
        points = np.append(points, new_points, axis=0)
        if len(points) < n:
            points = np.append(points, self._random_sampling_inside(n-len(points)),
                               axis=0)
        if len(points) > n:
            points = self._cut_points(points, n)
        return points.astype(np.float32)

    def _random_sampling_boundary(self, n):
        return super()._random_sampling_boundary(self.base, self.cut, n)

    def _grid_sampling_boundary(self, n):
        return super()._grid_sampling_boundary(self.base, self.cut, n)


class Union(Domain_operation):

    def __init__(self, domain_1, domain_2, n=100):
        self._check_correct_dim(domain_1, domain_2)
        self._check_if_input_is_interval(domain_1, domain_2)
        self.domain_1, self.domain_2 = self._order_domains_in_size(domain_1, domain_2)
        volume = self._approximate_volume(n)
        surface = self._approximate_surface(n)
        super().__init__(dim=domain_1.dim, tol=np.min((domain_1.tol, domain_2.tol)),
                         volume=volume, surface=surface)

    def _approximate_volume(self, n):
        volume_ratio = self._check_volume_ratio(self.domain_1, self.domain_2, n)
        return self.domain_2.volume+(1-volume_ratio)*self.domain_1.volume

    def _approximate_surface(self, n):
        surf_ratio_1 = self._check_boundary_ratio(self.domain_1, n)
        surf_ratio_2 = self._check_boundary_ratio(self.domain_2, n)
        return self.domain_1.surface*surf_ratio_1 + self.domain_2.surface*surf_ratio_2

    def _check_if_input_is_interval(self, domain_1, domain_2):
        if isinstance(domain_1, Interval) and isinstance(domain_2, Interval):
            if any(domain_1.is_inside(np.array(([domain_2.low_bound], 
                                                [domain_2.up_bound])))):
                raise ValueError('The intervals are not disjoint!') 
            if any(domain_2.is_inside(np.array(([domain_1.low_bound], 
                                                [domain_1.up_bound])))):
                raise ValueError('The intervals are not disjoint!')

    def _order_domains_in_size(self, domain_1, domain_2):
        if domain_1.volume > domain_2.volume:
            return domain_1, domain_2
        else:
            return domain_2, domain_1

    def is_inside(self, points):
        in_1 = self.domain_1.is_inside(points)
        in_2 = self.domain_2.is_inside(points)
        return np.logical_or(in_1, in_2) 

    def is_on_boundary(self, points):
        on_1 = self.domain_1.is_on_boundary(points)
        on_2 = self.domain_2.is_on_boundary(points)
        in_1 = self.domain_1.is_inside(points)
        in_2 = self.domain_2.is_inside(points)
        on_1 = np.logical_and(on_1, np.invert(in_2))
        on_2 = np.logical_and(on_2, np.invert(in_1))
        return np.logical_or(on_1, on_2)

    def boundary_normal(self, points):
        normals = np.zeros((len(points), self.dim))
        on_1 = self.domain_1.is_on_boundary(points)
        on_2 = self.domain_2.is_on_boundary(points)
        if any(on_1):
            index_1 = np.where(on_1)[0]
            normals[index_1] += self.domain_1.boundary_normal(points[index_1])
        if any(on_2):
            index_2 = np.where(on_2)[0]
            normals[index_2] += self.domain_2.boundary_normal(points[index_2])
        index_both = np.where(np.logical_and(on_1, on_2))[0]
        normals[index_both] *= 1/np.sqrt(2)  
        return normals.astype(np.float32)

    def grid_for_plots(self, n):
        n_1 = int(np.ceil(self.domain_1.volume/self.volume * n))
        points = self.domain_1.grid_for_plots(n_1)
        in_2 = self.domain_2.is_inside(points)
        index = np.where(np.invert(in_2))[0]
        points = points[index]
        n_2 = n - len(points)
        points = np.append(points, self.domain_2.grid_for_plots(n_2), axis=0)
        return points

    def _random_sampling_inside(self, n):
        return self._create_points_inside(n, type='random')

    def _grid_sampling_inside(self, n):
    	return self._create_points_inside(n, type='grid')

    def _create_points_inside(self, n, type):
        points = np.empty((0,self.dim))
        scaled_n = int(n*self.domain_1.volume/self.volume)
        new_points = self._sample_new_points_inside(self.domain_1, self.domain_2,
                                                    scaled_n, type=type)
        points = np.append(points, new_points, axis=0)
        scaled_n = n - len(points)
        new_points = self.domain_2.sample_inside(scaled_n, type=type)
        return np.append(points, new_points, axis=0).astype(np.float32)

    def _random_sampling_boundary(self, n):
        return super()._random_sampling_boundary(self.domain_1, self.domain_2, n)

    def _grid_sampling_boundary(self, n):
        return super()._grid_sampling_boundary(self.domain_1, self.domain_2, n)