"""Example script that trains a model to approximate
the solution of a 2D heat equation on the unit square
for time in [0, 1].
"""
import os
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

import time

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

device = 'cuda'

D = 1.18
temp_hot = 100.0
w, h = 10.0, 10.0

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
                             up_bound=3),
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
                                         num_workers=0))


def t_dirichlet_fun(input):
    return temp_hot*torch.sin(np.pi/w*input['x'][:, :1])*torch.sin(np.pi/h*input['x'][:, 1:])


t.add_train_condition(DirichletCondition(dirichlet_fun=t_dirichlet_fun,
                                         name='dirichlet',
                                         norm=norm,
                                         batch_size=1000,
                                         dataset_size=1000,
                                         num_workers=0,
                                         boundary_sampling_strategy='lower_bound_only'))


def pde(u, input):
    return gradient(u, input['t']) - D*laplacian(u, input['x'])


train_cond = DiffEqCondition(pde=pde,
                             norm=norm,
                             batch_size=2500,
                             dataset_size=2500,
                             num_workers=2)

setup = Setting(variables=(x, t),
                train_conditions={'pde': train_cond})

model = SimpleFCN(input_dim=3).to(device)

data = {}
conditions = setup.get_train_conditions()

for name in conditions:
    dataloader = conditions[name].get_dataloader()
    d = next(iter(dataloader))
    data[name] = d

for cn in data:
    for vn in data[cn]:
        data[cn][vn] = data[cn][vn].to(device)

optimizer = torch.optim.Adam(model.parameters())
iterations = 3000
start = time.time()
for i in range(iterations):
    optimizer.zero_grad()
    loss = torch.zeros(1, device=device, requires_grad=True)
    for name in conditions:
        d = data[name]
        # get error for this conditions
        c = conditions[name](model, d)
        # accumulate weighted error
        loss = loss + conditions[name].weight * c
    #print(loss)
    loss.backward()
    optimizer.step()
end = time.time()
print(end-start)

optimizer = torch.optim.LBFGS(model.parameters())
iterations = 3000
start = time.time()
for i in range(iterations):
    def closure():
        optimizer.zero_grad()
        loss = torch.zeros(1, device=device, requires_grad=True)
        for name in conditions:
            d = data[name]
            # get error for this conditions
            c = conditions[name](model, d)
            # accumulate weighted error
            loss = loss + conditions[name].weight * c
        #print(loss)
        loss.backward()
        return loss
    optimizer.step(closure)
end = time.time()
print(end-start)

loss = torch.zeros(1, device=device, requires_grad=True)
for name in conditions:
    d = data[name]
    # get error for this conditions
    c = conditions[name](model, d)
    # accumulate weighted error
    loss = loss + conditions[name].weight * c
print(loss)
"""
solver = PINNModule(model=SimpleFCN(input_dim=3),  # TODO: comput input_dim in setting
                    problem=setup)

trainer = pl.Trainer(gpus='-1',
                     logger=False,
                     num_sanity_val_steps=5,
                     checkpoint_callback=False,
                     benchmark=True)

trainer.fit(solver)
"""
