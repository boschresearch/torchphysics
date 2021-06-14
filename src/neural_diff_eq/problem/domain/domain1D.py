import numpy as np
import numbers

from .domain import Domain


class Interval(Domain):
    '''Class for intervals

    Parameters
    ----------
    low_bound : number
        The lower bound of the interval
    up_bound : number
        The upper bound of the interval
    tol : number, optional 
        The error tolerance for checking if points are inside or at the boundary 
    '''
    def __init__(self, low_bound, up_bound, tol=1e-06):
        if low_bound > up_bound:
            raise ValueError('The lower bound has to be smaller then the upper bound!')
        super().__init__(dim=1, volume=up_bound-low_bound,
                         surface=2, tol=tol)
        self.low_bound = low_bound
        self.up_bound = up_bound

    def is_inside(self, points):
        '''Checks if the given points are inside the open intervall
        
        Parameters
        ----------   
        points : number or array_like
            Either single point that has to be checked, or a list containing different
            points. The list has to be of the form [[x1],[x2],...]
        
        Returns 
        ----------
        np.array
            Every entry of the output contains either true, if the points was inside, 
            or false if not.  
        '''
        if isinstance(points, numbers.Number): points = np.array([points])
        return ((self.low_bound-self.tol < points[:]) 
                & (points[:] < self.up_bound+self.tol)).reshape(-1,1)

    def is_on_boundary(self, points):
        '''Checks if the given points are on the boundary of the intervall
        
        Parameters
        ----------   
        points : number or array_like
            Either single point that has to be checked, or a list containing different
            points. The list has to be of the form [[x1],[x2],...]
        
        Returns 
        ----------
        np.array
            Every entry of the output contains either true, if the points was inside,
            or false if not.  
        '''
        if isinstance(points, numbers.Number): points = np.array([points])
        return ((np.isclose(points[:], self.low_bound, atol=self.tol)) 
                | (np.isclose(points[:], self.up_bound, atol=self.tol))).reshape(-1, 1)
 
    def sample_boundary(self, n, type='random'):
        '''Samples points at the boundary of the domain

        Parameters
        ----------
        n : int
            Desired number of sample points
        type : {'random', 'grid', 'lower_bound_only', 'upper_bound_only'}
            The sampling strategy. 'random' and 'grid' are the same as described in the
            parent class
            - 'lower_bound_only' : Returns n times the lower bound
            - 'upper_bound_only' : Returns n times the upper bound 

        Returns
        -------
        np.array
            A array containing the points 
        '''   
        if type == 'lower_bound_only':
            return np.repeat(self.low_bound, n).astype(np.float32).reshape(-1, 1)
        elif type == 'upper_bound_only':
            return np.repeat(self.up_bound, n).astype(np.float32).reshape(-1, 1)
        else:
            return super().sample_boundary(n, type)
  
    def _random_sampling_inside(self, n):
        return np.random.uniform(self.low_bound,
            	                 self.up_bound,
                                 n).astype(np.float32).reshape(-1, 1)

    def _grid_sampling_inside(self, n):
        return np.linspace(self.low_bound, self.up_bound, 
                           n+2)[1:-1].astype(np.float32).reshape(-1, 1)

    def _random_sampling_boundary(self, n):
        return np.random.choice([self.low_bound, self.up_bound],
                                 n).astype(np.float32).reshape(-1, 1)

    def _grid_sampling_boundary(self, n):
        array_1 = np.repeat(self.low_bound, int(np.ceil(n/2)))
        array_2 = np.repeat(self.up_bound, int(np.floor(n/2)))
        return np.concatenate((array_1, array_2)).astype(np.float32).reshape(-1, 1)
    
    def grid_for_plots(self, n):
        return np.linspace(self.low_bound, 
                           self.up_bound, n).astype(np.float32).reshape(-1, 1)