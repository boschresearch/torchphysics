"""contains PyTorch models which can be trained to
approximate the solution of a differential equation"""

from .parameter import Parameter
from .model import (Model, NormalizationLayer, AdaptiveWeightLayer,
                    Sequential, Parallel)
from .fcn import FCN
from .deepritz import DeepRitzNet