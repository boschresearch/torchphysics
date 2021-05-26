"""Example script that trains a model to approximate
the solution of a 2D heat equation on the unit square
for time in [0, 1].
"""

from neural_diff_eq.problem import (Variable,
                                    Setting)
from neural_diff_eq.problem import domain
from neural_diff_eq.problem.domain import (Rectangle,
                                           Interval)
from neural_diff_eq.problem.condition import (DirichletCondition,
                                              DiffEqCondition)
from neural_diff_eq.models import SimpleFCN
from neural_diff_eq import Solver

x = Variable(name='x',
             order=2,
             domain=Rectangle(corner_dl=[0, 0],
                              corner_dr=[1, 0],
                              corner_tl=[0, 1]))
t = Variable(name='t',
             order=1,
             domain=Interval(low_bound=0,
                             up_bound=1))

x.add_train_condition(DirichletCondition())
