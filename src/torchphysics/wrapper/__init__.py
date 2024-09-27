""" Wrapper module that serves as a bridge between TorchPhysics and Nvidia Modulus and allows users to train TorchPhysics models with the Modulus training framework with minimal changes to their existing code. 
"""
from .wrapper import TPModulusWrapper
from .solver import ModulusSolverWrapper
from .model import ModulusArchitectureWrapper
from .helper import convertDataModulus2TP, convertDataTP2Modulus
    