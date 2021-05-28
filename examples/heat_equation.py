"""Example script that trains a model to approximate
the solution of a 2D heat equation on the unit square
for time in [0, 1].
"""
import torch
import numpy as np
import pytorch_lightning as pl

from neural_diff_eq.problem import (Variable,
                                    Setting)
from neural_diff_eq.problem.domain import (Rectangle,
                                           Interval)
from neural_diff_eq.problem.condition import (DirichletCondition,
                                              DiffEqCondition)
from neural_diff_eq.models import SimpleFCN
from neural_diff_eq import PINNModule
from neural_diff_eq.utils import laplacian, gradient

x = Variable(name='x',
             order=2,
             domain=Rectangle(corner_dl=[0, 0],
                              corner_dr=[1, 0],
                              corner_tl=[0, 1]),
             train_conditions={},
             val_conditions={})
t = Variable(name='t',
             order=1,
             domain=Interval(low_bound=0,
                             up_bound=1),
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
                                         num_workers=2))
x.add_val_condition(DirichletCondition(dirichlet_fun=x_dirichlet_fun,
                                       name='dirichlet',
                                       norm=norm,
                                       batch_size=500,
                                       dataset_size=500,
                                       num_workers=2))


def t_dirichlet_fun(input):
    return torch.sin(3.14*input['x'][:, :1])*torch.sin(3.14*input['x'][:, 1:])  # this may not work yet


t.add_train_condition(DirichletCondition(dirichlet_fun=t_dirichlet_fun,
                                         name='dirichlet',
                                         norm=norm,
                                         batch_size=500,
                                         dataset_size=500,
                                         num_workers=2,
                                         boundary_sampling_strategy='lower_bound_only'))
t.add_val_condition(DirichletCondition(dirichlet_fun=t_dirichlet_fun,
                                       name='dirichlet',
                                       norm=norm,
                                       batch_size=500,
                                       dataset_size=500,
                                       num_workers=2,
                                       boundary_sampling_strategy='lower_bound_only'))


def pde(u, input):
    return gradient(u, input['t']) - laplacian(u, input['x'])


train_cond = DiffEqCondition(pde=pde,
                             norm=norm,
                             batch_size=5000,
                             dataset_size=5000,
                             num_workers=2)
val_cond = DiffEqCondition(pde=pde,
                           norm=norm,
                           batch_size=5000,
                           dataset_size=5000,
                           num_workers=2)

setup = Setting(variables=(x, t),
                train_conditions={'pde': train_cond},
                val_conditions={'pde': val_cond})

solver = PINNModule(model=SimpleFCN(input_dim=3),  # TODO: comput input_dim in setting
                    problem=setup)

trainer = pl.Trainer(gpus=None,
                     logger=False,
                     num_sanity_val_steps=5,
                     checkpoint_callback=False)

trainer.fit(solver)
