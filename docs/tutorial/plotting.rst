========
Plotting
========
To present and evaluated the trained solutions, some helper functions are implemented. 
These can be found under ``torchphysics.utils``. We focus here on the **plot** method.
These method gets the following arguments:

- **model**: the model that got trained to learn a solution
- **plot_function**: a function that specifies what part of the models should be plotted
- **point_sampler**: a sampler that creates the points for the plot
- **plot_type**: what kind of plot should be created

Under the hood, the plotting uses the ``matplotlib.pyplot`` and essentially follows these steps:

1) Create points with the PointSampler
2) Evaluate the model
3) Apply the plot function
4) Try to find the correct type of plot, or use the specified type
5) Create the plot

In the following, the inputs are explained in more detail:

model
  Just an object of the ``Model``-class that was presented in the `model creation`_.

plot_function
  A function that can be defined like a condition function, see here_. For example, if 
  we just want to have a look on the output :math:`u` of the model, we use:

  .. code-block:: python 

    def plot_fn(u):
        return u # remeber u is the model outputs not the model itself 

  If we want to plot the laplcian w.r.t. :math:`x`, or any other kind of derivative:

  .. code-block:: python 

    def plot_fn(u, x):
        return tp.utils.laplcian(u, x)

  Or if we know the exact solution and want the error, we can compute this in there:
  
  .. code-block:: python 

    exact_sol = ......
    def plot_fn(u, x):
        return torch.abs(u - exact_sol(x))

point_sampler
  Here any sampler that is implemented can be used. But for plotting and animations some
  special samplers are also implemented, to make things easier. 
  These are the ``PlotSampler`` and ``AnimationSampler``.
  Since the plot will in general be created over a given domain, all other input
  variables of the model have to be constant values. 
  So both sampler get a single domain as an input and then create a point grid inside and 
  at the boundary. For all the other variables a dictionary can be passed as an input.

plot_type
  In general, the method picks the correct type depending on the output of the *plot_function*.
  But for two-dimensional domains, there are two different ways implemented. Either create
  a surface plot, *'surface_2D'*, or a contour plot, *'contour_surface'*.
  Default is the surface plot, if the other one should be used, that has to 
  specified.

The **plot** method just implements some basic presentations and is for complex domains **not**
perfectly optimized. Therefore, it should only be seen as a helper function for, a fast overview
of the solution.

The usage, on an explicit example, can be found under the tutorial: `solving a pde`_.


.. _`model creation`: model_creation.html
.. _here: condition_tutorial.html
.. _`solving a pde`: solve_pde.html