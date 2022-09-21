"""Contains the Spaces and Points classes.

Spaces 
  It's purpose is to define variable names and dimensions that can
  be used in functions, domains, models and more.

Points
  The ``Points`` object is a central part of TorchPhysics. They consit of a PyTorch-Tensor
  and a space. ``Points`` store data in a tensor with 2-axis, the first corresponding 
  the batch-dimension in a batch of multiple points. 
  The second axis collects the space dimensionalities.
"""

from .space import (Space,
                    R1, R2, R3, Rn)
from .points import Points
from .functionspace import FunctionSpace