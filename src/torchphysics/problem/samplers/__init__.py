"""Objects that sample points in a given domain. These objects handle the creation of
training and validation points in the underlying geometries. In general they get the following
inputs:

- **domain**: the domain in which the points should be created. If you want to create points
  at the boundary of a domain, use domain.boundary as an input argument.
- **n_points** or **density**: to set the number of wanted points. Either a fixed number can be
  chosen or the density of the points.
- **filter**: A function that filters out special points, for example for local boundary conditions.

The default behavior of each sampler is, that in each iteration of the trainings process new 
points are created and used. If this is not desired, not useful (grid sampling) or not
efficient (e.g. really complex domains) one can make every sampler ``static``.

Instead of creating the Cartesian product of different domains 
it is also possible to create the product of different samplers. For some sampling strategies, this
is required, e.g. point grids. 
If the product of the domains is created, the sampler will create points in the new domain. If the
product of samplers is created, first every sampler will create points in its own domain and afterwards 
the product will create a meshgrid of the points.
Therefore, the points of the product of samplers has in general more *correlation*, than the points
of the domain product.

"""

from .sampler_base import (PointSampler,
                           ProductSampler,
                           ConcatSampler,
                           AppendSampler,
                           StaticSampler,
                           EmptySampler)
from .random_samplers import (RandomUniformSampler,
                              GaussianSampler,
                              LHSSampler,
                              AdaptiveRandomRejectionSampler,
                              AdaptiveThresholdRejectionSampler)
from .grid_samplers import GridSampler, ExponentialIntervalSampler
from .plot_samplers import PlotSampler, AnimationSampler
from .data_samplers import DataSampler