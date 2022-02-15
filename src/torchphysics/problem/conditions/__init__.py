"""Conditions are the central concept in this package. They supply the necessary
training data to the model and translate the condition of the differential equation
into the trainings condition of the neural network.
"""

from .condition import (Condition,
                        PINNCondition,
                        DataCondition,
                        DeepRitzCondition,
                        ParameterCondition,
                        MeanCondition,
                        AdaptiveWeightsCondition,
                        SingleModuleCondition,
                        PeriodicCondition)