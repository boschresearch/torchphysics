#!/usr/bin/env python
# coding: utf-8

# # PINN: Heat equation with variable diffusion
# Solving the heat equation in 2D for variable diffusion D using the PINN-concept.

# In[1]:


import torch
import torchphysics as tp
import math


# First, we create the spaces for our problem. These define the variable names which will be used in the remaining part of this code.
# 
# In this example, x is the space variable, t corresponds to the time, D is an interval of diffusions and u is the variable for the (1D-)solution.

# In[2]:


X = tp.spaces.R2('x')
T = tp.spaces.R1('t')
D = tp.spaces.R1('D')

U = tp.spaces.R1('u')


# As a next step, we build the domain of the problem. There are multiple options to build multi-dimensional domains - in this case, we simply create a rectangle in space and intervals in time and diffusion which will later be multiplied to obtain the cartesian product.

# In[3]:


h, w = 20, 20


# In[4]:


A_x = tp.domains.Parallelogram(X, [0, 0], [w, 0], [0, h])
A_t = tp.domains.Interval(T, 0, 40)
A_D = tp.domains.Interval(D, 0.1, 1.0)


# Before we visualize the created domain, we create Sampler objects which are iterators that sample points from the domain during the optimization task. There are again various options to sample from the domains, an easy way would be to sample uniformly distributed random points. In this example, we choose an adaptive sampler to sample points in the inner of the domain. It will sample more points in points where the loss is large.
# 
# The amount of sampled points is defined by their density in the 3/2-dim subset, it could be increased to achieve better training results.

# In[5]:


inner_sampler = tp.samplers.AdaptiveRejectionSampler(A_x*A_t*A_D, density=1)
# initial values should be sampled on the left boundary of the time interval and for every x and D
initial_v_sampler = tp.samplers.RandomUniformSampler(A_x*A_t.boundary_left*A_D, density=1)

boundary_v_sampler = tp.samplers.RandomUniformSampler(A_x.boundary*A_t*A_D, density=1)


# We visualize the domain through the points created by the samplers using matplotlib:

# In[6]:


tp.utils.scatter(X*T, inner_sampler, initial_v_sampler, boundary_v_sampler)


# In the next step we define the NN-model we want to fit to the PDE. A normalization can improve convergence for large or small domains.

# In[7]:


model = tp.models.Sequential(
    tp.models.NormalizationLayer(A_x*A_t*A_D),
    tp.models.FCN(input_space=X*T*D, output_space=U, hidden=(50,50,50))
)


# Now, we define a condition which aims to minimze the mean squared error of the residual of the poisson equation. 

# In[8]:


def heat_residual(u, x, t, D):
    return D*tp.utils.laplacian(u, x) - tp.utils.grad(u, t)

pde_condition = tp.conditions.PINNCondition(module=model,
                                            sampler=inner_sampler,
                                            residual_fn=heat_residual,
                                            name='pde_condition')


# Additionally, we add a boundary condition at the boundary of the domain:

# In[9]:


def boundary_v_residual(u):
    return u

boundary_v_condition = tp.conditions.PINNCondition(module=model,
                                                     sampler=boundary_v_sampler,
                                                     residual_fn=boundary_v_residual,
                                                     name='boundary_condition')


# The initial condition can be defined via a data function. Again, we minimize the mean squared error over the sampled points.

# In[10]:


def f(x):
    return torch.sin(math.pi/w*x[:, :1])*torch.sin(math.pi/h*x[:,1:])

def initial_v_residual(u, f):
    return u-f

initial_v_condition = tp.conditions.PINNCondition(module=model,
                                                  sampler=initial_v_sampler,
                                                  residual_fn=initial_v_residual,
                                                  data_functions={'f': f},
                                                  name='initial_condition')


# For comparison, we compute the solution via a finite difference scheme.

# In[11]:


import sys
sys.path.append('..')

from fdm_heat_equation import FDM, transform_to_points

fdm_domain, fdm_time_domains, fdm_solution = FDM([0, w, 0, h], 2*[2e-1], [0,5], [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], f)
fdm_inp, fdm_out = transform_to_points(fdm_domain, fdm_time_domains, fdm_solution, [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], True)


# Comparsion to measured or computed data can be performed via a DataCondition using data supplied via a PointsDataLoader.

# In[12]:


val_condition = tp.conditions.DataCondition(module=model,
                                            dataloader=tp.utils.PointsDataLoader((fdm_inp, fdm_out), batch_size=80000),
                                            norm='inf')


# Finally, we optimize the conditions using a pytorch-lightning.LightningModule Solver and running the training. In the Solver, the training and validation conditions, as well as all optimizer options can be specified.

# In[14]:


solver = tp.solver.Solver([pde_condition,
                           boundary_v_condition,
                           initial_v_condition], [val_condition])

import pytorch_lightning as pl

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

trainer = pl.Trainer(gpus=1, # or None for CPU
                     max_steps=2000,
                     logger=False,
                     benchmark=True,
                     val_check_interval=400,
                     checkpoint_callback=False)
trainer.fit(solver)


# Finally, we plot the obtained solution:

# In[ ]:


anim_sampler = tp.samplers.AnimationSampler(A_x, A_t, 100, n_points=400, data_for_other_variables={'D': 1.0})
anim = tp.utils.animate(model, lambda u: u[:, 0], anim_sampler, ani_speed=10)
print("ready with all")

