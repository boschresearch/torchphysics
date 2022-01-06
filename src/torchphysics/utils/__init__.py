"""useful helper methods, e.g. metrics etc"""
from .differentialoperators import (laplacian,
                                    grad,
                                    div,
                                    jac,
                                    partial,
                                    convective,
                                    rot, 
                                    normal_derivative)

from .data import PointsDataset, PointsDataLoader

from .user_fun import UserFunction
from .plotting import plot, Plotter, animate, scatter
from .evaluation import compute_min_and_max