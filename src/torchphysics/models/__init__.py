"""Contains different PyTorch models which can be trained to
approximate the solution of a differential equation. 

Additional basic network structures are implemented, meant to stabilize and speed up
the trainings process. (adaptive weights, normalization layers)

If different models for different parts of the differential equation should be applied, this can be 
achieved by using the classes torchphysics.models.Sequential and torchphysics.models.Parallel.

Here you also find the parameters that can be learned in inverse problems.
"""

from .parameter import Parameter
from .model import (Model, NormalizationLayer, AdaptiveWeightLayer,
                    Sequential, Parallel)
from .fcn import FCN, Harmonic_FCN
from .deepritz import DeepRitzNet
from .qres import QRES
from .activation_fn import (AdaptiveActivationFunction, ReLUn, Sinus)

# DeepONet:
from .deeponet.deeponet import DeepONet
from .deeponet.branchnets import (BranchNet, FCBranchNet, ConvBranchNet1D)
from .deeponet.trunknets import (TrunkNet, FCTrunkNet) 
from .deeponet.layers import TrunkLinear