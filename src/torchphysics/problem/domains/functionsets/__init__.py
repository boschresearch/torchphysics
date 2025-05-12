"""
Function sets can be used to sample functions, e.g. in DeepONet.
"""

from .functionset import FunctionSet, DiscreteFunctionSet#, FunctionSetCollection, FunctionSetUnion
from .custom_functionset import CustomFunctionSet
from .harmonic_functionset import (HarmonicFunctionSet1D, 
                                   HarmonicFunctionSet2D,
                                   HarmonicFunctionSet3D)
