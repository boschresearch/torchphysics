import abc
import numpy as np


class Domain():
    '''Parent class for all domains.

    Parameters
    ----------
    dim : int
        The dimension of the domain.
    volume : float
        The "volume" of the domain. Stands for: 
            - 1D = length
            - 2D = area
            - 3D = volume 
    surface : float
        The "surface" area of the domain. Stands for:
            - 1D = boundary points (always 2)
            - 2D = perimeter
            - 3D = surface area              
    tol : number
        The error tolerance for checking if points are inside or at the boundary.
    '''

    def __init__(self, dim, volume, surface, tol):
        self.dim = dim
        self.volume = volume
        self.surface = surface
        self.tol = tol
   
    def sample_boundary(self, n, type='random'):
        '''Samples points at the boundary of the domain.

        Parameters
        ----------
        n : int
            Desired number of sample points.
        type : {'random', 'grid'}
            The sampling strategy. All child classes implement at least a random
            and a grid sampling. For additional strategies check the specific class.
            - 'random' : returns uniformly distributed points on the boundary
            - 'grid' : creates a grid over the boundary

        Returns
        -------
        np.array
            A array containing the points.
        '''
        if type == 'random':
            return self._random_sampling_boundary(n)
        elif type == 'grid':
            return self._grid_sampling_boundary(n)
        else:
            raise NotImplementedError

    def sample_inside(self, n, type='random'):
        '''Samples points in the inside of the domain

        Parameters
        ----------
        n : int
            Desired number of sample points
        type : {'random', 'grid'}
            The sampling strategy. All child classes implement at least a random
            and a grid sampling. For additional strategies check the specific class
            - 'random' : returns uniformly distributed points in the domain
            - 'grid' : creates a evenly grid over the domain.
                       Since it is not always possible to get a grid with excatly n pts
                       the center of the domain is added to get n points in total

        Returns
        -------
        np.array
            A array containing the points
        '''
        if type == 'random':
            return self._random_sampling_inside(n)
        elif type == 'grid':
            return self._grid_sampling_inside(n)
        else:
            raise NotImplementedError

    def _cut_points(self, n, points):
        """Cuts away some random points,
        if more than n were sampled (can happen by grid-sampling).
        """
        if len(points) > n:
            index = np.random.choice(len(points), int(n), replace=False)
            return points[index]
        return points

    def _check_inside_grid_enough_points(self, n, points):
        # checks if there are not enough points for the grid.
        # If not, add some random points 
        if len(points) < n:
            new_points = self._random_sampling_inside(n-len(points))
            points = np.append(points, new_points, axis=0)
        return points

    def _check_boundary_grid_enough_points(self, n, points):
        # checks if there are not enough points for the grid.
        # If not, add some random points 
        if len(points) < n:
            new_points = self._random_sampling_boundary(n-len(points))
            points = np.append(points, new_points, axis=0)
        return points

    @abc.abstractmethod
    def _compute_bounds(self):
        return 
        
    @abc.abstractmethod
    def is_inside(self, points):
        return

    @abc.abstractmethod
    def is_on_boundary(self, points):
        return

    @abc.abstractmethod
    def _random_sampling_inside(self, n):
        return

    @abc.abstractmethod
    def _grid_sampling_inside(self, n):
        return

    @abc.abstractmethod
    def _random_sampling_boundary(self, n):
        return

    @abc.abstractmethod
    def _grid_sampling_boundary(self, n):
        return

    @abc.abstractmethod
    def boundary_normal(self, points):
        return

    @abc.abstractmethod
    def grid_for_plots(self, n):
        return

    def serialize(self):
        dct = {}
        dct['dim'] = self.dim
        dct['tol'] = self.tol
        return dct
