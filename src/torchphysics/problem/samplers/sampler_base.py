"""The basic structure of every sampler and all sampler 'operations'.
"""
import abc
import torch
import warnings
import math 

from ...utils.user_fun import UserFunction
from ..spaces.points import Points


class PointSampler:
    """Handles the creation and interconnection of training/validation points.

    Parameters
    ----------
    n_points : int, optional
        The number of points that should be sampled.
    density : float, optional
        The desired density of the created points.
    filter_fn : callable, optional
        A function that restricts the possible positions of sample points.
        A point that is allowed should return True, therefore a point that should be 
        removed must return false. The filter has to be able to work with a batch
        of inputs.
        The Sampler will use a rejection sampling to find the right amount of points.
    """

    def __init__(self, n_points=None, density=None, filter_fn=None):
        self.n_points = n_points
        self.density = density
        self.length = None
        if filter_fn:
            self.filter_fn = UserFunction(filter_fn)
        else:
            self.filter_fn = None
    
    @classmethod
    def empty(cls, **kwargs):
        """Creates an empty Sampler object that samples empty points.

        Returns
        -------
        EmptySampler
            The empty sampler-object.
        """
        return EmptySampler().make_static()

    def set_length(self, length):
        """If a density is used, the number of points will not be known before
        hand. If len(PointSampler) is needed one can set the expected number 
        of points here.

        Parameters
        ----------
        length : int
            The expected number of points that this sampler will create.

        Notes
        -----
        If the domain is independent of other variables and a density is used, the 
        sampler will, after the first call to 'sample_points', set this value itself. 
        """
        self.length = length

    def __iter__(self):
        """Creates a iterator of this Pointsampler, with *next* the ``sample_points``
        methode can be called.
        """
        return self

    def __next__(self):
        return self.sample_points()

    def __len__(self):
        """Returns the number of points that the sampler will create or
        has created. 

        Note
        ----
        This can be only called if the number of points is set with ``n_points``.
        Elsewise the the number can only be known after the first call to 
        ``sample_points`` methode or may even change after each call.
        If you know the number of points yourself, you can set this with 
        ``.set_length``.
        """
        if self.length is not None:
            return self.length
        elif self.n_points is not None:
            return self.n_points
        else:
            raise ValueError("""The expected number of samples is not known yet. 
                                Set the length by using .set_length, if this 
                                property is needed""")

    def make_static(self, resample_interval =math.inf):
        """Transforms a sampler to an ``StaticSampler``. A StaticSampler only creates
        points the first time .sample_points() is called. Afterwards the points 
        are saved and will always be returned if .sample_points() is called again.
        Useful if the same points should be used while training/validation
        or if it is not practicall to create new points in each iteration
        (e.g. grid points).

        Parameters
        ----------
        resample_interval : int, optional
            Parameter to specify if new sampling of points should be created after a fixed number 
            of iterations. E.g. resample_interval =5, will use the same points for five iterations 
            and then sample a new batch that will be used for the next five iterations.
        """
        return StaticSampler(self, resample_interval)

    @property
    def is_static(self):
        """Checks if the Sampler is a ``StaticSampler``, e.g. retuns always the 
        same points.
        """
        return isinstance(self, StaticSampler)

    @property
    def is_adaptive(self):
        """Checks if the Sampler is a ``AdaptiveSampler``, e.g. samples points
        depending on the loss of the previous iteration.
        """
        return isinstance(self, AdaptiveSampler)

    def sample_points(self, params=Points.empty(), device='cpu'):
        """The method that creates the points. Also implemented in all child classes.

        Parameters
        ----------
        params : torchphysics.spaces.Points
            Additional parameters for the domain.
        device : str
            The device on which the points should be created.
            Default is 'cpu'.

        Returns
        -------
        Points:
            A Points-Object containing the created points and, if parameters were 
            passed as an input, the parameters. Whereby the input parameters
            will get repeated, so that each row of the tensor corresponds to  
            valid point in the given (product) domain.
        """
        if self.filter_fn:
            out = self._sample_points_with_filter(params, device)
        else:
            out = self._sample_points(params, device)
        return out

    @abc.abstractmethod
    def _sample_points_with_filter(self, params=Points.empty(), device='cpu'):
        raise NotImplementedError

    @abc.abstractmethod
    def _sample_points(self, params=Points.empty(), device='cpu'):
        raise NotImplementedError

    def __mul__(self, other):
        """Creates a sampler that samples from the 'Cartesian product'
        of the samples of two samplers, see ``ProductSampler``.
        """
        assert isinstance(other, PointSampler)
        return ProductSampler(self, other)

    def __add__(self, other):
        """Creates a sampler which samples from two different samples and 
        concatenates both outputs, see ``ConcatSampler``.
        """
        assert isinstance(other, PointSampler)
        return ConcatSampler(self, other)

    def append(self, other):
        """Creates a sampler which samples from two different samples and 
        makes a column stack of both outputs, see ``AppendSampler``.
        """
        assert isinstance(other, PointSampler)
        return AppendSampler(self, other)

    def _sample_params_independent(self, sample_function, params, device):
        """If the domain is independent of the used params it is more efficent
        to sample points once and then copy them accordingly.
        """
        points = sample_function(n=self.n_points, d=self.density, device=device)
        num_of_points = len(points)
        self.set_length(num_of_points)
        num_of_params = max(1, len(params))
        repeated_params = self._repeat_params(params, num_of_points)
        repeated_points = points.repeat(num_of_params)
        return repeated_points.join(repeated_params)

    def _sample_params_dependent(self, sample_function, params, device):
        """If the domain is dependent on some params, we can't always sample points
        for all params at once. Therefore we need a loop to iterate over the params.
        This happens for example with denstiy sampling or grid sampling. 
        """
        num_of_params = max(1, len(params))
        sample_points = None
        for i in range(num_of_params):
            new_points = self._sample_for_ith_param(sample_function, params, i, device)
            sample_points = self._set_sampled_points(sample_points, new_points)
        return sample_points

    def _sample_for_ith_param(self, sample_function, params, i, device):
        ith_params = params[i, ] if len(params) > 0 else Points.empty()
        new_points = sample_function(self.n_points, self.density, ith_params, device)
        num_of_points = len(new_points)
        repeated_params = self._repeat_params(ith_params, num_of_points)
        return new_points.join(repeated_params)

    def _set_sampled_points(self, sample_points, new_points):
        if not sample_points:
            return new_points
        return sample_points | new_points

    def _repeat_params(self, params, n):
        repeated_params = Points(torch.repeat_interleave(params, n, dim=0),
                                 params.space)
        return repeated_params

    def _apply_filter(self, sample_points):
        filter_true = self.filter_fn(sample_points)
        index = torch.where(filter_true)[0]
        return sample_points[index, ]

    def _check_iteration_number(self, iterations, num_of_new_points):
        if iterations == 10:
            warnings.warn(f"""Sampling points with filter did run 10
                              iterations and until now only found 
                              {num_of_new_points} from {self.n_points} points.
                              This may take some time.""")
        elif iterations >= 20 and num_of_new_points == 0:
            raise RuntimeError("""Run 20 iterations and could not find a single 
                                  valid point for the filter condition.""")

    def _cut_tensor_to_length_n(self, points):
        return points[:self.n_points, ]


class ProductSampler(PointSampler):
    """A sampler that constructs the product of two samplers.
    Will create a meshgrid (Cartesian product) of the data points of both samplers.

    Parameters
    ----------
    sampler_a, sampler_b : PointSampler
        The two PointSamplers that should be connected.
    """

    def __init__(self, sampler_a, sampler_b):
        self.sampler_a = sampler_a
        self.sampler_b = sampler_b
        super().__init__()

    def __len__(self):
        if self.length:
            return self.length
        return len(self.sampler_a) * len(self.sampler_b)

    def sample_points(self, params=Points.empty(), device='cpu'):
        b_points = self.sampler_b.sample_points(params, device=device)
        a_points = self.sampler_a.sample_points(b_points, device=device)
        self.set_length(len(a_points))
        return a_points


class ConcatSampler(PointSampler):
    """A sampler that adds two single samplers together.
    Will concatenate the data points of both samplers.

    Parameters
    ----------
    sampler_a, sampler_b : PointSampler
        The two PointSamplers that should be connected.
    """

    def __init__(self, sampler_a, sampler_b):
        self.sampler_a = sampler_a
        self.sampler_b = sampler_b
        super().__init__()

    def __len__(self):
        if self.length:
            return self.length
        return len(self.sampler_a) + len(self.sampler_b)

    def sample_points(self, params=Points.empty(), device='cpu'):
        samples_a = self.sampler_a.sample_points(params, device=device)
        samples_b = self.sampler_b.sample_points(params, device=device)
        self.set_length(len(samples_a) + len(samples_b))
        return samples_a | samples_b


class AppendSampler(PointSampler):
    """A sampler that appends the output of two samplers behind each other.
    Essentially calling torch.coloumn_stack for the data points.

    Parameters
    ----------
    sampler_a, sampler_b : PointSampler
        The two PointSamplers that should be connected. Both Samplers should create 
        the same number of points.
    """

    def __init__(self, sampler_a, sampler_b):
        self.sampler_a = sampler_a
        self.sampler_b = sampler_b
        super().__init__()

    def __len__(self):
        if self.length:
            return self.length
        return len(self.sampler_a)

    def sample_points(self, params=Points.empty(), device='cpu'):
        samples_a = self.sampler_a.sample_points(params, device=device)
        samples_b = self.sampler_b.sample_points(params, device=device)
        self.set_length(len(samples_a))
        return samples_a.join(samples_b)


class StaticSampler(PointSampler):
    """Constructs a sampler that saves the first points created and 
    afterwards only returns these points again. Has the advantage
    that the points only have to be computed once. Can also be customized to created new
    points after a fixed number of iterations.

    Parameters
    ----------
    sampler : Pointsampler
        The basic sampler that will create the points.  
    resample_interval : int, optional
        Parameter to specify if new sampling of points should be created after a fixed number 
        of iterations. E.g. resample_interval =5, will use the same points for five iterations 
        and then sample a new batch that will be used for the next five iterations.
    """

    def __init__(self, sampler, resample_interval=math.inf):
        self.length = None
        self.sampler = sampler
        self.created_points = None
        self.resample_interval = resample_interval 
        self.counter = 0

    def __len__(self):
        if self.length:
            return self.length
        return len(self.sampler)

    def __next__(self):
        if self.created_points:
            return self.created_points
        return self.sample_points()

    def sample_points(self, params=Points.empty(), device='cpu', **kwargs):
        self.counter += 1
        if self.created_points and self.counter < self.resample_interval:
            self._change_device(device=device)
            return self.created_points
        # reset counter if over self.resample_interval and create new points
        self.counter = 0
        points = self.sampler.sample_points(params, device=device, **kwargs)
        self.created_points = points
        return points

    def _change_device(self, device):
        self.created_points = self.created_points.to(device) 

    def make_static(self, resample_interval=math.inf):
        self.resample_interval = resample_interval
        return self


class EmptySampler(PointSampler):
    """A sampler that creates only empty Points. Can be used as a placeholder."""
    def __init__(self):
        super().__init__(n_points=0)
    
    def sample_points(self, params=Points.empty(), device='cpu', **kwargs):
        return Points.empty()
    



class AdaptiveSampler(PointSampler):
    """A sampler that requires a current loss for every point of the
    last sampled set of points.
    """

    def sample_points(self, unreduced_loss, params=Points.empty(), device='cpu'):
        """Extends the sample methode of the parent class. Also requieres the 
        unreduced loss of the previous iteration to create the new points.

        Parameters
        ----------
        unreduced_loss : torch.tensor
            The tensor containing the loss of each training point in the previous 
            iteration. 
        params : torchphysics.spaces.Points
            Additional parameters for the domain.
        device : str
            The device on which the points should be created.
            Default is 'cpu'.

        Returns
        -------
        Points:
            A Points-Object containing the created points and, if parameters were 
            passed as an input, the parameters. Whereby the input parameters
            will get repeated, so that each row of the tensor corresponds to  
            valid point in the given (product) domain.
        """
        if self.filter_fn:
            out = self._sample_points_with_filter(unreduced_loss=unreduced_loss,
                                                  params=params, device=device)
        else:
            out = self._sample_points(unreduced_loss=unreduced_loss,
                                      params=params, device=device)
        return out
