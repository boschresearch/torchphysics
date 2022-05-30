#!/usr/bin/env python
# coding: utf-8

# ### PINN approach for the 1-D duffing problem
# 
# $\ddot{x} + 2D\dot{x} + \omega_0^2(x + \mu x^3) = f(t)$

# In[9]:


import torch
import torchphysics as tp
import math
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

# In[19]:
omega_0 = 1

ode_params = {'D': .05,      # damping constant
              'c_nlin': 0.2  # nonlinearity
             }
sampling_frequency = 10 
t_end = 500 # sec
time = np.linspace(0, t_end, t_end*sampling_frequency + 1)
time_ind = pd.to_timedelta(time, unit="sec")


# ### ODE solution

# In[44]:


def duffing(t, y, D, c_nlin, F_func):
    z1, z2 = y
    dz1dt = z2
    dz2dt = - 2 * D * z2 - z1 - c_nlin * z1**3 + F_func(t)
    return(dz1dt, dz2dt)


# In[45]:
time.shape
F = pd.Series(np.random.randn(time.shape[0]),
              index=pd.Index(time, name="time")
              )# some white noise data on right hand site
F_inter_func = IUS(time, F, k=1)

x0 = [0, 0] # initial conditions
#%%

numeric_solu = solve_ivp(duffing,
                         t_span=(0, t_end),
                         y0 = x0,
                         max_step=1 / sampling_frequency,
                         args=(ode_params['D'], ode_params['c_nlin'], F_inter_func)
                        )

# In[ ]:
solu = pd.Series(numeric_solu.y[0,:], index=pd.Index(numeric_solu.t, name="time"))
solu.plot()

#%%
T = tp.spaces.R1('t') # time
D = tp.spaces.R1('D') # Damping 
C_nl = tp.spaces.R1('C_nl') # nonlinearity in stiffness
# Output
U = tp.spaces.R1('u')
# Intervals and domains and sampler
Dmin, Dmax = 0.01, 0.1
A_t = tp.domains.Interval(T, 0, t_end)
A_D = tp.domains.Interval(D, Dmin, Dmax)
A_cln = tp.domains.Interval(C_nl, 0, 0.5)
inner_sampler = (tp.samplers.GridSampler(A_t, n_points = 2500) \
                 *(tp.samplers.GridSampler(A_D, n_points = 10)\
                 *(tp.samplers.GridSampler(A_cln, n_points = 10)
                 ))).make_static()
    
initial_u_sampler = (tp.samplers.GridSampler(A_t.boundary_left, n_points = 1) \
                        *(tp.samplers.GridSampler(A_D, n_points = 10)\
                        *(tp.samplers.GridSampler(A_cln, n_points = 10)
                        ))).make_static()
initial_v_sampler = (tp.samplers.GridSampler(A_t.boundary_left, n_points = 1) \
                        *(tp.samplers.GridSampler(A_D, n_points = 10)\
                        *(tp.samplers.GridSampler(A_cln, n_points = 10)
                        ))).make_static()
    
# tp.utils.scatter(T * D * C_nl, inner_sampler)
#%% Model Definition
model = tp.models.Sequential(
    tp.models.NormalizationLayer(A_t * A_D * A_cln),
    tp.models.FCN(input_space=T * D * C_nl, output_space=U, hidden=(15,15)))

def duffing_residual(u, t, D, C_nl):
    return tp.utils.laplacian(u, t)  + 2 * D * tp.utils.grad(u, t) + u + C_nl * u**3 - F_inter_func(t)
pde_condition = tp.conditions.PINNCondition(module=model,
                                            sampler=inner_sampler,
                                            residual_fn=duffing_residual,
                                            name='pde_condition')

def initial_u_residual(u):
    return u
initial_u_condition = tp.conditions.PINNCondition(module=model,
                                                     sampler=initial_u_sampler,
                                                     residual_fn=initial_u_residual,
                                                     name='ic_u')

def initial_v_residual(u, t):
    return tp.utils.grad(u, t)
initial_v_condition = tp.conditions.PINNCondition(module=model,
                                                     sampler=initial_v_sampler,
                                                     residual_fn=initial_v_residual,
                                                     name='ic_v')

# def right_hand_side(t):
#     return F_inter_func(t)
# def total_residual(u, f):
#     return u-f
# force = tp.conditions.PINNCondition(module=model,
#                                     sampler=inner_sampler,
#                                     residual_fn=total_residual,
#                                     data_functions={'f': right_hand_side},
#                                     name='force')
#%% Training
opt_setting = tp.solver.OptimizerSetting(torch.optim.AdamW, lr=1e-2) #AdamW, SGD, LBFGS
solver = tp.solver.Solver([pde_condition,
                           initial_u_condition,
                           initial_v_condition], optimizer_setting = opt_setting)

trainer = pl.Trainer(gpus=None, # or 1 if GPU available
                     max_steps = 200,
                     #logger=False, 
                     benchmark=True,
                     log_every_n_steps=1,
                     )
trainer.fit(solver)









