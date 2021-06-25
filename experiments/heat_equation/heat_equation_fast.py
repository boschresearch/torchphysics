"""Example script that trains a model to approximate
the solution of a 2D heat equation on the unit square
for time in [0, 1].
"""
import os
from typing import Dict
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
from neural_diff_eq.datamodule import ProblemDataModule

import time

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

device = 'cuda'

D = 1.18
temp_hot = 100.0
w, h = 10.0, 10.0
print('--Create Problem--')
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
    return np.zeros_like(input['t'])


norm = torch.nn.MSELoss()


x.add_train_condition(DirichletCondition(dirichlet_fun=x_dirichlet_fun,
                                         name='dirichlet',
                                         norm=norm,
                                         batch_size=500,
                                         dataset_size=500,
                                         num_workers=0))


def t_dirichlet_fun(input):
    return temp_hot*np.sin(np.pi/w*input['x'][:, :1])*np.sin(np.pi/h*input['x'][:, 1:])


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

solver = PINNModule(model=model,
                    problem=setup,
                    optimizer=torch.optim.Adam,
                    lr=1e-3,
                    #log_plotter=plotter
                    )
datamod = ProblemDataModule(problem=setup, n_iterations=3000)
trainer = pl.Trainer(gpus='-1',
                     #accelerator='ddp',
                     #plugins=pl.plugins.DDPPlugin(find_unused_parameters=False),
                     num_sanity_val_steps=0,
                     check_val_every_n_epoch=100,
                     log_every_n_steps=1,
                     max_epochs=1,
                     # limit_val_batches=10,  # The validation dataset is probably pretty big,
                     # so you need to see how much you want to
                     # check every validation
                     # checkpoint_callback=False)
                     )

print('--Start Training--')
start = time.time()
trainer.fit(solver, datamod)
end = time.time()
print('Adam:', end-start)

solver.optimizer = torch.optim.LBFGS
start = time.time()
trainer.fit(solver, datamod)
end = time.time()
print('LBFGS:', end-start)
print('-- Prediction of Solution--')
resolution = 40
x = np.linspace(0, w, resolution)
y = np.linspace(0, h, resolution)
points = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
points = torch.FloatTensor(points).to(device)
time_points = 3 * torch.ones((1600, 1)).to(device)
input = {'x' : points, 't' : time_points}
start = time.time()
pred = solver(input).data.cpu().numpy()
end = time.time()
max_pred = np.max(pred)
min_pred = np.min(pred)
print('Time to evaluate model:', end-start)
print('Max at t =', time_points[0].item(), 'is:', max_pred)
print('Min at t =', time_points[0].item(), 'is:', min_pred)

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
