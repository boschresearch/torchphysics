"""useful helper methods, e.g. metrics etc"""
from .differentialoperators import (laplacian,
                                    grad,
                                    div,
                                    jac,
                                    partial,
                                    convective,
                                    normal_derivative)

from .helper import apply_to_batch, apply_user_fun
