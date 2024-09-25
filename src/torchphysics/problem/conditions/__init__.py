"""Conditions are the central concept in this package. They supply the necessary
training data to the model and translate the condition of the differential equation
into the trainings condition of the neural network.

A tutorial on the usage of Conditions can be found here_.

.. _here: https://boschresearch.github.io/torchphysics/tutorial/tutorial_start.html
"""

from .condition import (Condition,
                        PINNCondition,
                        DataCondition,
                        DeepRitzCondition,
                        ParameterCondition,
                        MeanCondition,
                        AdaptiveWeightsCondition,
                        SingleModuleCondition,
                        PeriodicCondition,
                        IntegroPINNCondition)

from .deeponet_condition import (DeepONetSingleModuleCondition, 
                                 PIDeepONetCondition, 
                                 DeepONetDataCondition)

from .variational_condition import VariationalPINNCondition