"""Conditions are the central concept in this package. They supply the necessary
training data to the model and translate the condition of the differential equation
into the trainings condition of the neural network.

A tutorial on the usage of Conditions can be found here_.

.. _here: https://torchphysics.readthedocs.io/en/latest/tutorial/condition_tutorial.html
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