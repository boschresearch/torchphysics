"""Samplers for plotting and animations of model outputs.
"""
import numpy as np
import torch

from ..domains.domain import BoundaryDomain
from ..domains import Interval
from .sampler_base import PointSampler
from .grid_samplers import GridSampler
from ..spaces.points import Points


class PlotSampler(PointSampler):
    """A sampler that creates a point grid over a domain
    (including the boundary). Only used for plotting,

    Parameters
    ----------
    plot_domain : Domain
        The domain over which the model/function should later be plotted.
        Will create points inside and at the boundary of the domain.
    n_points : int, optional
        The number of points that should be used for the plot.
    density : float, optional
        The desiered density of the created points.
    device : str or torch device, optional
        The device of the model/function.
    data_for_other_variables : dict or torchphysics.spaces.Points, optional
        Since the plot will only evaluate the model at a specific point, 
        the values for all other variables are needed. 
        E.g. {'t' : 1, 'D' : [1,2], ...}

    Notes
    -----
    Can also be used to create your own PlotSampler. By either changing the
    used sampler after the initialization (self.sampler=...) or by creating 
    your own class that inherits from PlotSampler.
    """
    def __init__(self, plot_domain, n_points=None, density=None, device='cpu',
                 data_for_other_variables={}):
        assert not isinstance(plot_domain, BoundaryDomain), \
            "Plotting for boundaries is not implemented"""
        super().__init__(n_points=n_points, density=density)
        self.device = device
        self.created_points = None
        self.set_data_for_other_variables(data_for_other_variables)
        self.domain = plot_domain(**self.data_for_other_variables.coordinates)
        self.sampler = self.construct_sampler()

    def set_data_for_other_variables(self, data_for_other_variables):
        """Sets the data for all other variables. Essentially copies the
        values into a correct tensor.
        """
        if isinstance(data_for_other_variables, Points):
            self.data_for_other_variables = data_for_other_variables
        elif len(data_for_other_variables) == 0:
            self.data_for_other_variables = Points.empty()
        else:
            torch_data = self.transform_data_to_torch(data_for_other_variables)
            self.data_for_other_variables = Points.from_coordinates(torch_data)

    def transform_data_to_torch(self, data_for_other_variables):
        """Transforms all inputs to a torch.tensor.
        """
        torch_data = {}
        for vname, data in data_for_other_variables.items():
            # transform data to torch
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data)
            # check correct shape of data
            if len(data.shape) == 0:
                torch_data[vname] = data.reshape(-1, 1)
            elif len(data.shape) == 1:
                torch_data[vname] = data.reshape(-1, len(data))
            else:
                torch_data[vname] = data
        return torch_data

    def construct_sampler(self):
        """Construct the sampler which is used in the plot.
        Can be overwritten to include your own points structure.
        """
        if self.n_points:
            return self._plot_sampler_with_n_points()
        else: # density is used
            return self._plot_sampler_with_density()

    def _plot_sampler_with_n_points(self):
        if isinstance(self.domain, Interval):
            return self._construct_sampler_for_Interval(self.domain, n=self.n_points)
        inner_n_points = self._compute_inner_number_of_points()
        inner_sampler = GridSampler(self.domain, inner_n_points)
        outer_sampler = GridSampler(self.domain.boundary, len(self)-inner_n_points)
        return inner_sampler + outer_sampler

    def _plot_sampler_with_density(self):
        if isinstance(self.domain, Interval):
            return self._construct_sampler_for_Interval(self.domain, d=self.density)
        inner_sampler = GridSampler(self.domain, density=self.density)
        outer_sampler = GridSampler(self.domain.boundary, density=self.density)
        return inner_sampler + outer_sampler

    def _construct_sampler_for_Interval(self, domain, n=None, d=None):
        left_sampler = GridSampler(domain.boundary_left, 1)
        inner_sampler = GridSampler(domain, n_points=n, density=d)
        right_sampler = GridSampler(domain.boundary_right, 1)
        return left_sampler + inner_sampler + right_sampler       

    def _compute_inner_number_of_points(self):
        n_root = int(np.ceil(len(self)**(1/self.domain.dim)))
        n_root -= 2
        return n_root**self.domain.dim

    def sample_points(self, params=Points.empty(), device='cpu'):
        """Creates the points for the plot. Does not need additional arguments, since
        they were set in the init.
        """
        if not self.created_points:
            self.device = device
            plot_points = self.sampler.sample_points(device=device)
            self.set_length(len(plot_points))
            self.data_for_other_variables = self.data_for_other_variables.to(device)
            other_data = self._repeat_params(self.data_for_other_variables, len(self))
            self.created_points = plot_points.join(other_data)
        self._set_device_and_grad_true()
        return self.created_points

    def _set_device_and_grad_true(self):
        self.created_points._t.requires_grad = True
        self.created_points._t.to(self.device)


class AnimationSampler(PlotSampler):
    """A sampler that creates points for an animation.

    Parameters
    ----------
    plot_domain : Domain
        The domain over which the model/function should later be plotted.
        Will create points inside and at the boundary of the domain.
    animation_domain : Interval
        The variable over which the animation should be created, e.g a 
        time-interval.
    frame_number : int
        The number of frames that should be used for the animation. This
        equals the number of points that will be created in the 
        animation_domain.
    n_points : int, optional
        The number of points that should be used for the plot domain.
    density : float, optional
        The desiered density of the created points, in the plot domain.
    device : str or torch device, optional
        The device of the model/function.
    data_for_other_variables : dict, optional
        Since the animation will only evaluate the model at specific points, 
        the values for all other variables are needed. 
        E.g. {'D' : [1,2], ...}
    """
    def __init__(self, plot_domain, animation_domain, frame_number, 
                 n_points=None, density=None, device='cpu',
                 data_for_other_variables={}):
        super().__init__(plot_domain=plot_domain, n_points=n_points,
                         density=density, device=device,
                         data_for_other_variables=data_for_other_variables)
        self._check_correct_types(animation_domain)
        self.frame_number = frame_number
        self.animation_domain = animation_domain(**data_for_other_variables)
        self.animatoin_sampler = \
            self._construct_sampler_for_Interval(self.animation_domain, n=frame_number)

    def _check_correct_types(self, animation_domain):
        assert isinstance(animation_domain, Interval), \
            "The animation domain has to be a interval"

    @property
    def plot_domain_constant(self):
        """Returns if the plot domain is a constant domain or changes
        with respect to other variables.
        """
        dependent = any(vname in self.domain.necessary_variables \
                        for vname in self.animation_domain.space)
        return not dependent

    @property
    def animation_key(self):
        """Retunrs the name of the animation variable
        """
        ani_key = list(self.animation_domain.space.keys())[0]
        return ani_key 

    def sample_animation_points(self):
        """Samples points out of the animation domain, e.g. time interval.
        """
        ani_points = self.animatoin_sampler.sample_points()
        num_of_points = len(ani_points)
        self.frame_number = num_of_points
        self._set_device_and_grad_true(ani_points)
        return ani_points

    def sample_plot_domain_points(self, animation_points):
        """Samples points in the plot domain, e.g. space.
        """
        if self.plot_domain_constant:
            plot_points = self.sampler.sample_points()
            num_of_points = len(plot_points)
            self.set_length(num_of_points)
            self._set_device_and_grad_true(plot_points)
            return plot_points
        return self._sample_params_dependent(animation_points)

    def _sample_params_dependent(self, params):
        output_list = []
        for i in range(self.frame_number):
            ith_ani_points = params[i, ]
            plot_points = self.sampler.sample_points(ith_ani_points)
            plot_points._t.to(self.device)
            output_list.append(plot_points)
        return output_list

    def _set_device_and_grad_true(self, p):
        p._t.requires_grad = True
        p._t.to(self.device)