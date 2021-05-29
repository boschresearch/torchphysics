"""Example script to check how good the model
can approximate solutions with different thermal conductivity D
"""
import os

import torch
import numpy as np
import pytorch_lightning as pl
from timeit import default_timer as timer

from neural_diff_eq.problem import (Variable,
                                    Setting)
from neural_diff_eq.problem.domain import (Rectangle,
                                           Interval)
from neural_diff_eq.problem.condition import (DirichletCondition,
                                              DiffEqCondition,
                                              DataCondition)
from neural_diff_eq.models import SimpleFCN
from neural_diff_eq import PINNModule
from neural_diff_eq.utils import laplacian, gradient
from neural_diff_eq.utils.fdm import FDM, create_validation_data

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

w, h = 50, 50
t0, tend = 0, 1
temp_hot = 10
D_low, D_up = 5, 25  # set here the interval boundary for D

x = Variable(name='x',
             order=2,
             domain=Rectangle(corner_dl=[0, 0],
                              corner_dr=[w, 0],
                              corner_tl=[0, h]),
             train_conditions={},
             val_conditions={})
t = Variable(name='t',
             order=1,
             domain=Interval(low_bound=0,
                             up_bound=tend),
             train_conditions={},
             val_conditions={})
D = Variable(name='D',
             order=0,
             domain=Interval(low_bound=D_low,
                             up_bound=D_up),
             train_conditions={},
             val_conditions={})


def x_dirichlet_fun(input):
    return torch.zeros_like(input['t'])


norm = torch.nn.MSELoss()


x.add_train_condition(DirichletCondition(dirichlet_fun=x_dirichlet_fun,
                                         name='dirichlet',
                                         norm=norm,
                                         batch_size=500,
                                         dataset_size=500,
                                         num_workers=4))


def t_dirichlet_fun(input):
    return temp_hot*torch.sin(np.pi/w*input['x'][:, :1])*torch.sin(np.pi/h*input['x'][:, 1:])


t.add_train_condition(DirichletCondition(dirichlet_fun=t_dirichlet_fun,
                                         name='dirichlet',
                                         norm=norm,
                                         batch_size=500,
                                         dataset_size=500,
                                         num_workers=4,
                                         boundary_sampling_strategy='lower_bound_only'))


def pde(u, input):
    return gradient(u, input['t']) - input['D']*laplacian(u, input['x'])


train_cond = DiffEqCondition(pde=pde,
                             norm=norm,
                             batch_size=5000,
                             dataset_size=5000,
                             num_workers=8)

# FDM:
domain_dic = {'x': [[0, w], [0, h]]}
dx, dy = 0.5, 0.5
step_width_dict = {'x': [dx, dy]}
time_interval = [t0, tend]


def inital_condition(input):
    return temp_hot * np.sin(np.pi/w*input['x'][:, :1]) * np.sin(np.pi/h*input['x'][:, 1:])


D_list = [5, 10, 15, 20, 25]
# ^Here you can add many different values for D, e.g [18.8,2.5,20,....]
# The FDM-Methode will compute solutions for all D.
# For too many D this will become really memory expensive, since
# the FDM uses a forward euler!
fdm_start = timer()
domain, time, u = FDM(domain_dic, step_width_dict, time_interval,
                      D_list, inital_condition)
fdm_end = timer()
print('Time for FDM-Solution:', fdm_end-fdm_start)
data_x, data_u = create_validation_data(domain, time, u, D_list, D_is_input=True)
# True: if D is input of the model

val_cond = DataCondition(data_x=data_x,
                         data_u=data_u,
                         name='validation',
                         norm=norm,
                         batch_size=len(data_u[:, 0])//100,
                         num_workers=16)

setup = Setting(variables=(x, t, D),
                train_conditions={'pde': train_cond},
                val_conditions={'validation': val_cond})

solver = PINNModule(model=SimpleFCN(input_dim=4),  # TODO: comput input_dim in setting
                    problem=setup)

trainer = pl.Trainer(gpus='-1',
                     #logger=False,
                     num_sanity_val_steps=2,
                     check_val_every_n_epoch=100,
                     log_every_n_steps=1,
                     # limit_val_batches=10,  # The validation dataset is probably pretty big,
                     # so you need to see how much you want to
                     # check every validation
                     checkpoint_callback=False)

trainer.fit(solver)
