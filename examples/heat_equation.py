"""Example script that trains a model to approximate
the solution of a 2D heat equation on the unit square
for time in [0, 1].
"""
import torch

from neural_diff_eq.problem import (Variable,
                                    Setting)
from neural_diff_eq.problem import domain
from neural_diff_eq.problem.domain import (Rectangle,
                                           Interval)
from neural_diff_eq.problem.condition import (DirichletCondition,
                                              DiffEqCondition)
from neural_diff_eq.models import SimpleFCN
from neural_diff_eq import PINNModule

x = Variable(name='x',
             order=2,
             domain=Rectangle(corner_dl=[0, 0],
                              corner_dr=[1, 0],
                              corner_tl=[0, 1]))
t = Variable(name='t',
             order=1,
             domain=Interval(low_bound=0,
                             up_bound=1))


def x_dirichlet_fun(input):
    return torch.zeros_like(input['t'])


x.add_train_condition(DirichletCondition(dirichlet_fun=x_dirichlet_fun,
                                         name='dirichlet'))
x.add_val_condition(DirichletCondition(dirichlet_fun=x_dirichlet_fun,
                                         name='dirichlet'))


def t_dirichlet_fun(input):
    return torch.sin(input['x'][0])  # this may not work yet


t.add_train_condition(DirichletCondition(dirichlet_fun=t_dirichlet_fun,
                                         name='dirichlet'))
t.add_val_condition(DirichletCondition(dirichlet_fun=t_dirichlet_fun,
                                         name='dirichlet'))


def pde(u, input):
    pass


pde_cond = DiffEqCondition(pde=pde)

setup = Setting(train_conditions={'pde': pde_cond})
