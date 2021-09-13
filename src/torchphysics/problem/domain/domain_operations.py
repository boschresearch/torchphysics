import numpy as np
import abc
from .domain import Domain
from .domain1D import Interval
from .domain2D import Rectangle, Circle, Triangle, Polygon2D


class Domain_operation(Domain):
    '''Parent class for all domain operations.
    '''
    def __init__(self, dim, volume, surface, tol):
        super().__init__(dim, volume, surface, tol)

    def _check_correct_dim(self, domain_1, domain_2):
        if domain_1.dim != domain_2.dim:
            raise ValueError('Domains have to be of the same dimension,' \
                             ' found dimensions %d and %d' %(domain_1.dim, 
                             domain_2.dim))

    def _check_if_input_is_interval(self, domain):
        if isinstance(domain, Interval):
            raise ValueError('It is more efficient to create a new interval!')

    def _check_boundary_ratio(self, domain, n=100, type='grid'):
        '''Approximates how much percent of the boundary from the old domain 
        belongs to the new domain.

        Parameters
        ----------
        domain : Domain
            Original part of the new united/cut/intersected domain.
        n : int, optional
            Number of points that should be used to compute the ratio. 
            More Points equals a better approximation.
        type : str, optional
            Type of sampling on the boundary. 
        '''
        points = domain.sample_boundary(n, type)
        number = len(points[np.where(self.is_on_boundary(points))[0]])
        return number/n

    def _check_volume_ratio(self, domain_1, domain_2, n=100, type='grid'):
        '''Approximates how much percent of the inside from the old domain 
        belongs to the new domain.

        Parameters
        ----------
        domain_1, domain_2 : Domain
            Original parts of the new united/cut/intersected domain.
        n : int, optional
            Number of points that should be used to compute the ratio. 
            More Points equals a better approximation.
        type : str, optional
            Type of the inside sampling. 
        '''
        points = domain_1.sample_inside(n, type)
        number = len(points[np.where(domain_2.is_inside(points))[0]])
        return number/n

    def _sample_new_points_inside(self, domain_1, domain_2, n, type):
        new_points = domain_1.sample_inside(n, type) # sample points
        inside_2 = domain_2.is_inside(new_points) # check if there are inside of
                                                  # of the new domain    
        index = np.where(inside_2)[0]
        new_points = np.delete(new_points, index, axis=0)
        return new_points

    def _lhs_sampling_inside(self, n):
        if self.dim == 1: # Union of two intervals
            n_1 = int(n * self.domain_1.volume/self.volume)
            points = self.domain_1._lhs_sampling_inside(n_1)
            points_2 = self.domain_2._lhs_sampling_inside(n-n_1)
            return np.append(points, points_2, axis=0)
        elif self.dim == 2:
            return Triangle._grid_in_triangle(self, n, type='lhs')
        else:
            raise NotImplementedError

    def _sample_new_points_boundary(self, domain_1, n, type):
        new_points = domain_1.sample_boundary(n, type) # sample points
        on_bound = self.is_on_boundary(new_points) # check if there at the boundary of
                                                   # of the new domain
        index = np.where(np.invert(on_bound))[0]
        new_points = np.delete(new_points, index, axis=0)
        return new_points

    def _random_sampling_boundary(self, domain_1, domain_2, n):
        points = np.empty((0,self.dim))
        domains = [domain_1, domain_2]
        scaled_n = [int(np.ceil(domain_1.surface*n/self.surface)),
                    int(np.ceil(domain_2.surface*n/self.surface))]
        current_domain_is_1 = True
        # alternate between the two domains and sample on each boundary
        while len(points) < n:
            current_domain = domains[current_domain_is_1]
            current_n = scaled_n[current_domain_is_1]
            new_points = self._sample_new_points_boundary(current_domain, current_n,
                                                          type='random')
            points = np.append(points, new_points, axis=0)
            current_domain_is_1 = not current_domain_is_1
        points = super()._cut_points(n, points)
        return points.astype(np.float32)

    def _grid_sampling_boundary(self, domain_1, domain_2, n):
        # sample on domain_1, scale the n according to the percent of the surface
        n_1 = int(domain_1.surface*n/self.surface)
        points = self._sample_new_points_boundary(domain_1, n_1, type='grid')
        # sample on domain_2, scale the n according to the percent of the surface
        n_2 = int(domain_2.surface*n/self.surface)
        new_points = self._sample_new_points_boundary(domain_2, n_2, type='grid')
        points = np.append(points, new_points, axis=0)
        points = super()._check_boundary_grid_enough_points(n, points)
        points = super()._cut_points(n, points)
        return points.astype(np.float32)

    def _normal_sampling_boundary(self, domain_1, domain_2, n, mean, cov):
        on_1 = domain_1.is_on_boundary([mean])
        on_2 = domain_2.is_on_boundary([mean])
        if on_1 & on_2: # divide n and sample n/2 points on each domain
            points = self._sample_normal_points(domain_1,
                                                int(n/2), mean, cov)
            points_2 = self._sample_normal_points(domain_2,
                                                  n-len(points), 
                                                  mean, cov)
            return np.append(points, points_2, axis=0)
        else:
            mean_domain = domain_1
            other_domain = domain_2
            if on_2:
                mean_domain = domain_2
                other_domain = domain_1
            return self._mean_only_on_one_domain(mean_domain, other_domain,
                                                 n, mean, cov)

    def _sample_normal_points(self, domain, n, mean, cov):
        # samples points only on one domain and checks if they are and the
        # new domain
        points = np.empty((0, self.dim))
        while len(points) < n:
            new_points = domain._normal_sampling_boundary(n-len(points), mean, cov)
            on_bound = self.is_on_boundary(new_points)
            ind = np.where(on_bound)[0]
            points = np.append(points, new_points[ind], axis=0)
        return points.astype(np.float32)

    def _mean_only_on_one_domain(self, mean_domain, other_domain, n, mean, cov):
        # first try to sample on the domain where the mean is.
        # Maybe all created points lay in the connected domain -> done
        points = mean_domain._normal_sampling_boundary(n, mean, cov)
        on_bound = self.is_on_boundary(points)
        if all(on_bound):
            return points
        else:
            points = points[np.where(on_bound)[0]]
            missing = n - len(points)
            mean = self._approx_mean_on_other_domain(other_domain, mean, n)
            new_points = self._sample_normal_points(other_domain, missing, 
                                                    mean, cov)
        return np.append(points, new_points, axis=0)

    def _approx_mean_on_other_domain(self, domain, point, n):
        grid = domain._grid_sampling_boundary(n)
        on_bound = np.where(self.is_on_boundary(grid))[0]
        grid = grid[on_bound]
        norm = np.linalg.norm(grid - point, axis=1)
        return grid[np.argmin(norm)]

    def _get_boundary_normal(self, domain_1, domain_2, points, operation_is_cut=False):
        normals = np.zeros((len(points), self.dim))
        on_1 = domain_1.is_on_boundary(points)
        on_2 = domain_2.is_on_boundary(points)
        points = np.array(points)
        if any(on_1): # points at boundary of domain_1
            index_1 = np.where(on_1)[0]
            normals[index_1] += domain_1.boundary_normal(points[index_1])
        if any(on_2): # points at boundary of domain_2
            index_2 = np.where(on_2)[0]
            if operation_is_cut:
                # normals of the cut domain needs to be multiplied by -1
                normals[index_2] -= domain_2.boundary_normal(points[index_2])
            else : 
                normals[index_2] += domain_2.boundary_normal(points[index_2])
        index_both = np.where(np.logical_and(on_1, on_2))[0]
        normals[index_both] *= 1/np.sqrt(2) # point at both -> scale vector
        return normals.astype(np.float32)

    def construct_shapely(self, domain):
        """Construct the given domain as an shapely polygon,
        to show the outline. 

        Paramters
        ---------
        domain : Domain
            The input domain, for which a shapely object will be constructed.

        Returns 
        -------
        shapely.geometry.polygon
            The domain as a shapely polygon
        """
        import shapely.geometry as s_geos
        if isinstance(domain, Domain_operation):
            return domain._change_to_shapely()
        elif isinstance(domain, Rectangle):
            return s_geos.Polygon([domain.corner_dl, domain.corner_dr, 
                                  domain.corner_dr+(domain.corner_tl-domain.corner_dl), 
                                  domain.corner_tl, domain.corner_dl])
        elif isinstance(domain, Circle):
            return s_geos.Point(domain.center).buffer(domain.radius)
        elif isinstance(domain, Triangle):
            return s_geos.Polygon(domain.corners) 
        elif isinstance(domain, Polygon2D):
            return domain.polygon
        else:
            raise NotImplementedError

    @abc.abstractmethod
    def _approximate_volume(self, n):
        return

    @abc.abstractmethod
    def _approximate_surface(self, n):
        return

    def serialize(self):
        return super().serialize()


class Cut(Domain_operation):
    '''Implements the operation: A\B.

    Parameters
    ----------
    base : Domain
        The base domain, of which the other domain should be cut off.    
    cut : Domain 
        The domain that should be cut off from the base
    n : int, optional
        Number of points that should be used to approximate the volume and surface
        of the cut domain.
    '''
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
        # Instead of exactly computing the volume we only approximate it. 
        # Needed for example if we want to cut this domain again. 
        volume_ratio = self._check_volume_ratio(self.cut, self.base, n)
        return self.base.volume-volume_ratio*self.cut.volume
        
    def _approximate_surface(self, n):
        # Instead of exactly computing the surface we only approximate it. 
        # Needed for example if we want to cut this domain again. 
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
        return super()._get_boundary_normal(self.base, self.cut, points,
                                            operation_is_cut=True)

    def grid_for_plots(self, n):
        scaled_n = int(7/8*self.base.volume/self.volume * n)
        points = self.base.grid_for_plots(scaled_n)
        inside = self.cut.is_inside(points)
        index = np.where(inside)[0]
        points = np.delete(points, index, axis=0)
        # add some points at the boundary to better show the domain
        points_boundary = self.cut._grid_sampling_boundary(int(np.ceil(n/8)))
        inside = self.base.is_inside(points_boundary)
        bound = self.base.is_on_boundary(points_boundary)
        index = np.where(np.logical_or(np.invert(inside), bound))[0]
        points_boundary = np.delete(points_boundary, index, axis=0)
        return np.append(points, points_boundary, axis = 0)
  
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
        points = super()._check_inside_grid_enough_points(n, points)
        points = super()._cut_points(n, points)
        return points.astype(np.float32)

    def _random_sampling_boundary(self, n):
        return super()._random_sampling_boundary(self.base, self.cut, n)

    def _grid_sampling_boundary(self, n):
        return super()._grid_sampling_boundary(self.base, self.cut, n)

    def _normal_sampling_boundary(self, n, mean, cov):
        return super()._normal_sampling_boundary(self.base, self.cut, n, mean, cov)

    def serialize(self):
        # to show data/information in tensorboard
        dct = super().serialize()
        dct_1 = self.base.serialize()
        dct_2 = self.cut.serialize()
        dct['name'] = '(' + dct_1['name'] + ' - ' + dct_2['name'] + ')'
        dct['base'] = dct_1
        dct['cut'] = dct_2
        return dct

    def _compute_bounds(self):
        """computes bounds of the domain

        Returns
        -------
        np.array:
            The bounds in the form: [min_x, max_x, min_y, max_y]
        """
        return self.base._compute_bounds()

    def outline(self):
        """Creates a outline of the domain.

        Returns
        -------
        shapely.geometry.polygon
            A polygon, that contains the form of this domain.
        """
        domain = self._change_to_shapely()
        cords = [np.array(domain.exterior.coords)] 
        for i in domain.interiors:
            cords.append(np.array(i.coords))
        return cords 

    def _change_to_shapely(self):
        """Implements the specific operation (cut)
        """
        base = self.construct_shapely(self.base)
        cut = self.construct_shapely(self.cut)
        return base - cut


class Union(Domain_operation):
    '''Implements the Union of two domains

    Parameters
    ----------
    domain_1, domain_2 : Domain
        The two domains who should be united.    
    n : int, optional
        Number of points that should be used to approximate the volume and surface
        of the cut domain.
    '''  
    def __init__(self, domain_1, domain_2, n=100):
        self._check_correct_dim(domain_1, domain_2)
        self._check_if_input_is_interval(domain_1, domain_2)
        self.domain_1, self.domain_2 = self._order_domains_in_size(domain_1, domain_2)
        volume = self._approximate_volume(n)
        surface = self._approximate_surface(n)
        super().__init__(dim=domain_1.dim, tol=np.min((domain_1.tol, domain_2.tol)),
                         volume=volume, surface=surface)

    def _approximate_volume(self, n):
        # Instead of exactly computing the volume we only approximate it. 
        # Needed for example if we want cut/unit/intersect this domain again. 
        volume_ratio = self._check_volume_ratio(self.domain_1, self.domain_2, n)
        return self.domain_2.volume+(1-volume_ratio)*self.domain_1.volume

    def _approximate_surface(self, n):
        # Instead of exactly computing the surface we only approximate it. 
        # Needed for example if we want cut this domain again. 
        surf_ratio_1 = self._check_boundary_ratio(self.domain_1, n)
        surf_ratio_2 = self._check_boundary_ratio(self.domain_2, n)
        return self.domain_1.surface*surf_ratio_1 + self.domain_2.surface*surf_ratio_2

    def _check_if_input_is_interval(self, domain_1, domain_2):
        # It would be okay to unite two disjoint intervals 
        if isinstance(domain_1, Interval) and isinstance(domain_2, Interval):
            if any(domain_1.is_inside(np.array(([domain_2.low_bound], 
                                                [domain_2.up_bound])))):
                raise ValueError('The intervals are not disjoint!') 

    def _order_domains_in_size(self, domain_1, domain_2):
        # the order is important for a good distribution of points
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
        return super()._get_boundary_normal(self.domain_1, self.domain_2, points)

    def grid_for_plots(self, n):
        # create gird in first domain:
        scaled_n = int(n*self.domain_1.volume/self.volume)
        points = self.domain_1.grid_for_plots(scaled_n)
        inside = self.domain_2.is_inside(points)
        index = np.where(inside)[0]
        points = np.delete(points, index, axis=0)
        # create points on second domain:
        scaled_n = n - len(points)
        new_points = self.domain_2.grid_for_plots(scaled_n)
        points = np.append(points, new_points, axis=0)
        return points.astype(np.float32)

    def sample_boundary(self, n, type='random', sample_params={}):
        # check if we have intervals, for them we have special strategies
        if self.dim == 1:
            if type == 'lower_bound_only':
                low_bound = np.min([self.domain_1.low_bound, self.domain_2.low_bound])
                return np.repeat(low_bound, n).astype(np.float32).reshape(-1, 1)
            elif type == 'upper_bound_only':
                up_bound = np.max([self.domain_1.up_bound, self.domain_2.up_bound])
                return np.repeat(up_bound, n).astype(np.float32).reshape(-1, 1)
        return super().sample_boundary(n, type, sample_params)

    def _random_sampling_inside(self, n):
        return self._create_points_inside(n, type='random')

    def _grid_sampling_inside(self, n):
    	return self._create_points_inside(n, type='grid')

    def _create_points_inside(self, n, type):
        # just create points in either domain
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

    def _normal_sampling_boundary(self, n, mean, cov):
        return super()._normal_sampling_boundary(self.domain_1, self.domain_2,
                                                 n, mean, cov)

    def serialize(self):
        # to show data/information in tensorboard
        dct = super().serialize()
        dct_1 = self.domain_1.serialize()
        dct_2 = self.domain_2.serialize()
        dct['name'] = '(' + dct_1['name'] + ' + ' + dct_2['name'] + ')'
        dct['domain 1'] = dct_1
        dct['domain 2'] = dct_2
        return dct

    def _compute_bounds(self):
        """computes bounds of the domain

        Returns
        -------
        np.array:
            The bounds in the form: [min_x, max_x, min_y, max_y]
        """
        b_1 = self.domain_1._compute_bounds()
        b_2 = self.domain_2._compute_bounds()
        return [np.min([b_1[0], b_2[0]]), np.max([b_1[1], b_2[1]]), 
                np.min([b_1[2], b_2[2]]), np.max([b_1[3], b_2[3]])]

    def outline(self):
        """Creates a outline of the domain.

        Returns
        -------
        shapely.geometry.polygon
            A polygon, that contains the form of this domain.
        """
        domain = self._change_to_shapely()
        cords = [np.array(domain.exterior.coords)] 
        for i in domain.interiors:
            cords.append(np.array(i.coords))
        return cords 

    def _change_to_shapely(self):
        from shapely.ops import unary_union
        """Implements the specific operation (union)
        """
        domain_1 = self.construct_shapely(self.domain_1)
        domain_2 = self.construct_shapely(self.domain_2)
        return unary_union([domain_1, domain_2])


class Intersection(Domain_operation):
    '''Implements the intersection of two domains.

    Parameters
    ----------
    domain_1, domain_2 : Domain
        The two domains who should be intersected.    
    n : int, optional
        Number of points that should be used to approximate the volume and surface
        of the cut domain.
    '''  
    def __init__(self, domain_1, domain_2, n=100):
        self._check_correct_dim(domain_1, domain_2)
        self._check_if_input_is_interval(domain_1)
        self.domain_1, self.domain_2 = self._order_domains_in_size(domain_1, domain_2)
        volume = self._approximate_volume(n)
        surface = self._approximate_surface(n)
        super().__init__(dim=domain_1.dim, tol=np.min((domain_1.tol, domain_2.tol)),
                         volume=volume, surface=surface)

    def _order_domains_in_size(self, domain_1, domain_2):
        if domain_1.surface > domain_2.surface:
            return domain_1, domain_2
        else:
            return domain_2, domain_1

    def _approximate_volume(self, n):
        # Instead of exactly computing the volume we only approximate it. 
        # Needed for example if we want cut/unit/intersect this domain again.
        volume_ratio = self._check_volume_ratio(self.domain_1, self.domain_2, n)
        return volume_ratio*self.domain_1.volume

    def _approximate_surface(self, n):
        # Instead of exactly computing the volume we only approximate it. 
        # Needed for example if we want cut/unit/intersect this domain again.
        surf_ratio_1 = self._check_boundary_ratio(self.domain_1, n)
        surf_ratio_2 = self._check_boundary_ratio(self.domain_2, n)
        return self.domain_1.surface*surf_ratio_1 + self.domain_2.surface*surf_ratio_2

    def is_inside(self, points):
        in_1 = self.domain_1.is_inside(points)
        in_2 = self.domain_2.is_inside(points)
        return np.logical_and(in_1, in_2)

    def is_on_boundary(self, points):
        in_1 = self.domain_1.is_inside(points)
        in_2 = self.domain_2.is_inside(points)
        on_1 = self.domain_1.is_on_boundary(points)
        on_2 = self.domain_2.is_on_boundary(points)
        return np.logical_or(np.logical_and(in_1, on_2), np.logical_and(in_2, on_1))

    def boundary_normal(self, points):
        return super()._get_boundary_normal(self.domain_1, self.domain_2, points)

    def _random_sampling_inside(self, n):
        points = np.empty((0, self.dim))
        while n > 0:
            new_points = self.domain_1._random_sampling_inside(n)
            in_2 = self.domain_2.is_inside(new_points)
            index = np.where(in_2)[0]
            points = np.append(points, new_points[index], axis=0)
            n -= len(new_points[index])
        return points

    def _grid_sampling_inside(self, n):
        scaled_n = int(self.domain_1.volume/self.volume * n)
        points = np.empty((0,self.dim))
        new_points = self.domain_1._grid_sampling_inside(scaled_n)
        in_2 = self.domain_2.is_inside(new_points)
        index = np.where(in_2)[0]
        points = np.append(points, new_points[index], axis=0)
        points = super()._check_inside_grid_enough_points(n, points)
        points = super()._cut_points(n, points)
        return points.astype(np.float32)

    def _random_sampling_boundary(self, n):
        return super()._random_sampling_boundary(self.domain_1, self.domain_2, n)

    def _grid_sampling_boundary(self,n):
        return super()._grid_sampling_boundary(self.domain_1, self.domain_2, n)

    def _normal_sampling_boundary(self, n, mean, cov):
        return super()._normal_sampling_boundary(self.domain_1, self.domain_2,
                                                 n, mean, cov)

    def serialize(self):
        # to show data/information in tensorboard
        dct = super().serialize()
        dct_1 = self.domain_1.serialize()
        dct_2 = self.domain_2.serialize()
        dct['name'] = '(' + dct_1['name'] + ' intersect ' + dct_2['name'] + ')'
        dct['domain 1'] = dct_1
        dct['domain 2'] = dct_2
        return dct

    def grid_for_plots(self, n):
        # create gird in first domain:
        scaled_n = int(3/4*n*self.domain_1.volume/self.volume)
        points = self.domain_1.grid_for_plots(scaled_n)
        inside = self.domain_2.is_inside(points)
        index = np.where(np.invert(inside))[0]
        points = np.delete(points, index, axis=0)
        # create points on boundary
        points_boundary = self._grid_sampling_boundary(int(np.ceil(n/4)))
        return np.append(points, points_boundary, axis=0).astype(np.float32)

    def _compute_bounds(self):
        """computes bounds of the domain

        Returns
        -------
        np.array:
            The bounds in the form: [min_x, max_x, min_y, max_y]
        """
        b_1 = self.domain_1._compute_bounds()
        b_2 = self.domain_2._compute_bounds()
        return [np.max([b_1[0], b_2[0]]), np.min([b_1[1], b_2[1]]), 
                np.max([b_1[2], b_2[2]]), np.min([b_1[3], b_2[3]])]

    def outline(self):
        """Creates a outline of the domain.

        Returns
        -------
        shapely.geometry.polygon
            A polygon, that contains the form of this domain.
        """
        domain = self._change_to_shapely()
        cords = [np.array(domain.exterior.coords)] 
        for i in domain.interiors:
            cords.append(np.array(i.coords))
        return cords 

    def _change_to_shapely(self):
        """Implements the specific operation (intersection)
        """
        domain_1 = self.construct_shapely(self.domain_1)
        domain_2 = self.construct_shapely(self.domain_2)
        return domain_1 & domain_2