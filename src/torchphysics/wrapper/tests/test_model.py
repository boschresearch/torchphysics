import pytest
import shutil
import os
import torch
from torchphysics.wrapper.model import ModulusArchitectureWrapper

from torchphysics.problem.spaces import Space
from torchphysics.problem.spaces.points import Points

@pytest.fixture
def input_space():
    return Space({'x': 2})

@pytest.fixture
def output_space():
    return Space({'output': 1})

@pytest.fixture
def modulus_architecture_wrapper(input_space, output_space):
    return ModulusArchitectureWrapper(input_space, output_space, arch_name="fully_connected")

def test_modulus_architecture_wrapper_init(modulus_architecture_wrapper):
    assert isinstance(modulus_architecture_wrapper, ModulusArchitectureWrapper)
    assert list(modulus_architecture_wrapper.input_space.keys()) == ['x']
    assert list(modulus_architecture_wrapper.output_space.keys()) == ['output']  
    assert modulus_architecture_wrapper.input_space.dim == 2
    assert modulus_architecture_wrapper.output_space.dim == 1
    assert type(modulus_architecture_wrapper.modulus_net).__name__=='FullyConnectedArch'

def test_modulus_architecture_wrapper_forward(modulus_architecture_wrapper):
    in_vars = Points.from_coordinates({'x': torch.tensor([[1.0, 2.0], [3.0, 4.0]])})
    result = modulus_architecture_wrapper.forward(in_vars)
    assert isinstance(result, Points)
    assert list(result.coordinates.keys())==['output']
    assert result.coordinates['output'].shape == torch.Size([2,1])


def teardown_module(module):
    """This method is called after test completion."""
    conf_dir = os.path.abspath(os.getcwd() + '/conf')
    if os.path.isdir(conf_dir):
        shutil.rmtree(conf_dir)