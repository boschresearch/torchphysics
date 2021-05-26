import abc
import numpy as np


class Domain():
    '''Parent class for all domains

    Parameters
    ----------
    dim : int
        The dimension of the domain
    tol : number
        The error toleranz for checking if points are inside or at the boundary
    '''

    def __init__(self, dim, tol):
        self.dim = dim
        self.tol = tol

    def sample_boundary(self, n, type='random'):
        '''Samples points at the boundary of the domain

        Parameters
        ----------
        n : int
            Desired number of sample points
        type : {'random', 'grid'}
            The sampling strategy. All child classes implement at least a random
            and a grid sampling. For additional strategies check the specific class

        Returns
        -------
        np.array
            A array containing the points
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

    def vector_normalize(self, vector, order=2):
        norm = np.linalg.norm(vector, order)
        return vector/norm

    @abc.abstractmethod
    def is_inside(self, x):
        return

    @abc.abstractmethod
    def is_on_boundary(self, x):
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
    def boundary_normal(self, x):
        return
