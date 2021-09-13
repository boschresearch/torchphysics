import abc
import numpy as np
import numbers

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
   
    def sample_boundary(self, n, type='random', sample_params=None):
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
                - 'normal' : Creates a normal distribution of points.
                             Needs the additional inputs 'mean' and 'cov', for the 
                             center and variance of the distribution. 
                             The mean has to be a point on the boundary, while
                             the variance-matirx has to be one dimension smaller
                             then the domain.
        sample_params : dict
            A dictionary containing additional parameters for specific stragtegies.
            E.g. for the normal distribution the 'mean' and (co-)variance. Then
            the input would for example be (for a 2D-Domain):
            sample_params = {'mean': [0, 1], 'cov': 0.5} 
            For more information check the wanted methode to see the needed inputs. 
            
        Returns
        -------
        np.array
            A array containing the points.
        '''
        if type == 'random':
            return self._random_sampling_boundary(n)
        elif type == 'grid':
            return self._grid_sampling_boundary(n)
        elif type == 'normal':
            return self._normal_sampling_boundary(n, sample_params['mean'], 
                                                  sample_params['cov'])
        else:
            raise NotImplementedError

    def sample_inside(self, n, type='random', sample_params=None):
        '''Samples points inside of the domain

        Parameters
        ----------
        n : int
            Desired number of sample points
        type : {'random', 'grid', 'normal'}
            The sampling strategy. All child classes implement at least a random,
            grid, normal and lhs sampling.
            For additional strategies check the specific class.
            - 'random' : Returns uniformly distributed points in the domain
            - 'grid' : Creates a evenly grid over the domain.
                       Since it is not always possible to get a grid with excatly 
                       n points, additional random points will be added.
            - 'normal' : Creates a normal distribution of points.
                         Needs the additional inputs 'mean' and 'cov', for the 
                         center and variance of the distribution. The inputs need to
                         fit the dimension of the domain.
            - 'lhs' : Creates a latin hypercube sampling inside the domain.
        sample_params : dict
            A dictionary containing additional parameters for specific stragtegies.
            E.g. for the normal distribution the 'mean' and (co-)variance. Then
            the input would for example be (for a 2D-Domain):
            sample_params = {'mean': [0, 1], 'cov': [[1 0], [0, 1]]} 
            For more information check the wanted methode to see the needed inputs. 
        Returns
        -------
        np.array
            A array containing the points
        '''
        if type == 'random':
            return self._random_sampling_inside(n)
        elif type == 'grid':
            return self._grid_sampling_inside(n)
        elif type == 'normal':
            return self._normal_sampling_inside(n, sample_params['mean'], 
                                                sample_params['cov'])
        elif type == 'lhs':
            return self._lhs_sampling_inside(n)
        else:
            raise NotImplementedError

    def _normal_sampling_inside(self, n, mean, cov):
        """Uses a rejection sampling to create points with a
        normal distribution.

        Parameters
        ----------
        n : int
            Desired number of points.
        mean : list or array
            The center/mean of the distribution. 
        cov : number, list or array
            The (co-)variance of the distribution. For dimensions >= 2,
            cov has to be a symmetric and positive-semidefinite matrix.
            If a number is given as an input and the dimension is not 
            1, the covariance will be set to cov*unit-matrix.

        Returns
        -------
        np.array
            A array containing the points
        """
        if isinstance(cov, numbers.Number):
            cov = cov * np.eye(self.dim)
        points = np.empty((0, self.dim))
        while len(points) < n:
            new_points = np.random.multivariate_normal(mean, cov, size=n-len(points))
            inside = self.is_inside(new_points)
            points = np.append(points, new_points[np.where(inside)[0]], axis=0)
        return points.astype(np.float32)

    def _lhs_sampling_inside(self, n):
        lhs_axis = np.empty((n, self.dim))
        # divide each axis and compute lhs
        for i in range(self.dim):
            divide_axis = np.linspace(0, 1, n, endpoint=False)
            random_points = np.random.uniform(0, 1/n, n)
            new_axis = np.add(divide_axis, random_points)
            lhs_axis[:, i] = np.random.permutation(new_axis)
        return lhs_axis.astype(np.float32)

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
    def _normal_sampling_boundary(self, n, mean, cov):
        """Uses a rejection sampling to create points with a
        normal distribution.

        Parameters
        ----------
        n : int
            Desired number of points.
        mean : list or array
            The center/mean of the distribution. 
        cov : number, list or array
            The (co-)variance of the distribution. For dimensions n >= 3,
            cov has to be a symmetric and positive-semidefinite matrix of dim.
            n-1. If a number is given as an input and the dimension is not 
            2, the covariance will be set to cov*unit-matrix.

        Returns
        -------
        np.array
            A array containing the points
        """
        raise NotImplementedError


    @abc.abstractmethod
    def boundary_normal(self, points):
        """Computes the boundary normal.

        Parameters
        ----------
        points : list of lists
            A list containing all points where the normal vector has to be computed,
            e.g. in 2D: [[x1,y1],[x2,y2],...].

        Returns
        ----------
        np.array
            Every entry of the output contains the normal vector at the point,
            specified in the input array.
        """
        return

    @abc.abstractmethod
    def grid_for_plots(self, n):
        return

    def serialize(self):
        dct = {}
        dct['dim'] = self.dim
        dct['tol'] = self.tol
        return dct