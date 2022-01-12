====================
Solving a simple PDE
====================
The following problem also exists as a notebook_. Here we assume that you know all the
basic concepts that were part of the previous tutorials and will only give short explanations
to every step.

.. _notebook: notebooks.solving_pde.ipynb

Our aim is to solve the following PDE:

.. math::

   \begin{align}
   \Delta u &= 4.25\pi^2 u \text{ in } \Omega = [0, 1] \times [0, 1] \\
   u &= \sin(\frac{\pi}{2} x_1)\cos(2\pi x_2) \text{ on } \partial \Omega
   \end{align}

For comparison, the analytic solution is $u(x_1, x_2) = \sin(\frac{\pi}{2} x_1)\cos(2\pi x_2)$.

We start by defining the spaces for the input and output values.

.. code-block:: python

    import torchphysics as tp 
    X = tp.spaces.R2('x') # input is 2D
    U = tp.spaces.R1('u') # output is 1D

Next up is the domain:

.. code-block:: python

    square = tp.domains.Parallelogram(X, [0, 0], [1, 0], [0, 1])


Now we define our model, that we want to train. Since we have a simple domain, we do not use any 
normalization.

.. code-block:: python

    model = tp.models.FCN(input_space=X, output_space=U, hidden=(50,50,50,50,50))

The next step is the definition of the conditions. For this PDE we have two different ones, the 
differential equation itself and the boundary condition. We start with the boundary condition:

.. code-block:: python

    import torch
    import numpy as np
    # Frist the function that defines the residual:
    def bound_residual(u, x):
        bound_values = torch.sin(np.pi/2*x[:, :1]) * torch.cos(2*np.pi*x[:, 1:])
        return u - bound_values

    # the point sampler, for the trainig points:
    # here we use grid points any other sampler could also be used
    bound_sampler = tp.samplers.GridSampler(square.boundary, n_points=5000)
    bound_sampler = bound_sampler.make_static() # grid always the same, therfore static to one single computation
    # wrap everything together in the condition
    bound_cond = tp.conditions.PINNCondition(module=model, sampler=bound_sampler, 
                                             residual_fn=bound_residual, weight=10)

It follows the differential condition, here we use the pre implemented operators:

.. code-block:: python

    # Again a function that defines the residual:
    def pde_residual(u, x):
        return tp.utils.laplacian(u, x) + 4.25*np.pi**2*u

    # the point sampler, for the trainig points:
    pde_sampler = tp.samplers.GridSampler(square, n_points=15000) # again point grid 
    pde_sampler = pde_sampler.make_static()
    # wrap everything together in the condition
    pde_cond = tp.conditions.PINNCondition(module=model, sampler=pde_sampler, 
                                           residual_fn=pde_residual)


The transformation of our PDE into a TorchPhysics problem is finished. So we can start the
training.

The last step before the training is the creation of a *Solver*. This is an object that inherits from
the Pytorch Lightning *LightningModule*. It handles the training and validation loops and takes care of 
the data loading for GPUs or CPUs. It gets the following inputs:

- train_conditions: A list of all train conditions
- val_conditions: A list of all validation conditions (optional)
- optimizer_setting: With this, one can specify what optimizers, learning, and learning-schedulers 
  should be used. For this, there exists the class *OptimizerSetting* that handles all these parameters.

.. code-block:: python

    # here we start with Adam:
    optim = tp.OptimizerSetting(optimizer_class=torch.optim.Adam, lr=0.001)

    solver = tp.solver.Solver(train_conditions=[bound_cond, pde_cond], optimizer_setting=optim)

Now we define the trainer, for this we use Pytorch Lightning. Almost all functionalities of
Pytorch Lightning can be applied in the trainings process.

.. code-block:: python

    import pytorch_lightning as pl
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" # select GPUs to use
    trainer = pl.Trainer(gpus=1, # or None if CPU is used
                         max_steps=4000, # number of training steps
                         logger=False,
                         benchmark=True,
                         checkpoint_callback=False)
                        
    trainer.fit(solver) # start training

Afterwards we switch to LBFGS:

.. code-block:: python

    optim = tp.OptimizerSetting(optimizer_class=torch.optim.LBFGS, lr=0.05, 
                                optimizer_args={'max_iter': 2, 'history_size': 100})

    solver = tp.solver.Solver(train_conditions=[bound_cond, pde_cond], optimizer_setting=optim)

    trainer = pl.Trainer(gpus=1,
                         max_steps=3000, # number of training steps
                         logger=False,
                         benchmark=True,
                         checkpoint_callback=False)
                        
    trainer.fit(solver)

If we want to have a look on our solution, we can use the plot-methods of TorchPhysics:

.. code-block:: python 

    plot_sampler = tp.samplers.PlotSampler(plot_domain=square, n_points=600, device='cuda')
    fig = tp.utils.plot(model, lambda u : u, plot_sampler, plot_type='contour_surface')

.. image:: pictures/solution.png
  :width: 400
  :align: center
  :alt: solution of the PDE

We can plot the error, since we know the exact solution:

.. code-block:: python 

    def plot_fn(u, x):
        exact = torch.sin(np.pi/2*x[:, :1])*torch.cos(2*np.pi*x[:, 1:])
        return torch.abs(u - exact)
    fig = tp.utils.plot(model, plot_fn, plot_sampler, plot_type='contour_surface')

.. image:: pictures/error.png
  :width: 400
  :align: center
  :alt: error of the solution

Now you know how to solve a PDE in TorchPhysics, additional examples can 
be found under the `example-folder`_.

.. _`example-folder`: https://github.com/boschresearch/torchphysics/tree/main/examples