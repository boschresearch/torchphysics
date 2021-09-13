import torch
import numpy as np
import pytest
import torchphysics.models as models


def test_prepare_input():
    Dmodel = models.fcn.DiffEqModel(variable_dims={'x': 2, 't': 1}, 
                                    solution_dims={'u': 2})
    input = {'x': torch.ones((2, 2)), 't': torch.zeros((2, 1))}
    out = Dmodel._prepare_inputs(input)
    assert torch.equal(out, torch.FloatTensor([[1, 1, 0], [1, 1, 0]]))
    input = {'x': torch.ones((2, 2))}
    out = Dmodel._prepare_inputs(input)
    assert torch.equal(out, torch.FloatTensor([[1, 1], [1, 1]]))


def test_prepare_wrong_input_dim(capfd):
    Dmodel = models.fcn.DiffEqModel(variable_dims={'x': 2, 't': 1}, 
                                    solution_dims={'u': 2})
    input = {'x': torch.ones((2, 3)), 't':  torch.ones((2, 1))}
    Dmodel._prepare_inputs(input)
    out, _ = capfd.readouterr()
    assert out == """The input x has the wrong dimension. This can
                              lead to unexpected behaviour.\n"""


def test_prepare_too_many_inputs(capfd):
    Dmodel = models.fcn.DiffEqModel(variable_dims={'x': 2, 't': 1}, 
                                        solution_dims={'u': 2})
    input = {'x': torch.ones((2, 2)), 't':  torch.ones((2, 1)),
             'D': torch.ones((3, 3))}
    Dmodel._prepare_inputs(input)
    out, _ = capfd.readouterr()
    assert out == """The given variable names do not fit variable_dims.
                      This can lead to unexpected behaviour.
                      Please use Variables ['x', 't'].\n"""

def test_serialize_diffeqmodel():
    Dmodel = models.fcn.DiffEqModel(variable_dims={'x': 2, 't': 1}, 
                                    solution_dims={'u': 2})
    with pytest.raises(NotImplementedError):
        Dmodel.serialize()


def test_normalization_layer():
    Dmodel = models.fcn.DiffEqModel(variable_dims={'x': 2, 't': 1}, 
                                    solution_dims={'u': 2}, 
                                    normalization_dict={'x': [[2, 2], [9, 2]], 
                                                        't': [4, 0]})
    assert isinstance(Dmodel.normalize, torch.nn.Linear)
    assert Dmodel.normalize.weight[0][0] == 1
    assert Dmodel.normalize.weight[1][1] == 1
    assert Dmodel.normalize.weight[2][2] == 1/2
    assert Dmodel.normalize.bias[0] == -9
    assert Dmodel.normalize.bias[1] == -2
    assert Dmodel.normalize.bias[2] == 0


# Test SimpleFCN:
def _create_fcn(in_dim={'x': 2, 't':1}):
    fcn = models.fcn.SimpleFCN(variable_dims=in_dim, 
                               solution_dims={'u': 1},
                               depth=1, width=10)
                               
    return fcn 

def test_create_simpleFCN():
    fcn = models.fcn.SimpleFCN(variable_dims={'x': 2, 't': 1}, 
                               solution_dims={'u': 2},
                               depth=2, width=10)
    assert fcn.input_dim == 3
    assert fcn.width == 10
    assert fcn.depth == 2
    assert fcn.output_dim == 2


def test_structur_of_simpleFCN():
    fcn = models.fcn.SimpleFCN(variable_dims={'x': 2}, 
                               solution_dims={'u': 1},
                               depth=3, width=10)
    assert isinstance(fcn.layers, torch.nn.ModuleList)
    for i in range(5):
        assert isinstance(fcn.layers[2*i], torch.nn.Linear)
    for i in range(4):
        assert isinstance(fcn.layers[2*i+1], torch.nn.Tanh)
    assert fcn.layers[0].in_features == 2
    assert fcn.layers[0].out_features == 10
    for i in range(1,4):
        assert fcn.layers[2*i].in_features == 10
        assert fcn.layers[2*i].out_features == 10
    assert fcn.layers[8].in_features == 10
    assert fcn.layers[8].out_features == 1


def test_serialize_simpleFCN():
    fcn = models.fcn.SimpleFCN(variable_dims={'x': 2}, 
                               solution_dims={'u': 1})
    dic = fcn.serialize()
    assert dic['name'] == 'SimpleFCN'
    assert dic['input_dim'] == 2
    assert dic['depth'] == 3
    assert dic['width'] == 20
    assert dic['output_dim'] == 1


def test_forward_simpleFCN():
    fcn = _create_fcn()
    input = {'x': torch.ones((2, 2)), 't': torch.zeros((2, 1))}
    output = fcn.forward(input)
    assert output['u'].shape == (2, 1)
    assert isinstance(output['u'], torch.Tensor)
    assert isinstance(output, dict)


def test_get_layers_of_simpleFCN():
    fcn = _create_fcn()
    layers = fcn.get_layers()
    assert isinstance(layers, torch.nn.ModuleList)
    for i in range(0, 5, 2):
        assert isinstance(layers[i], torch.nn.Linear)


def test_change_weights_of_layer_with_float():
    fcn = _create_fcn()
    fcn.set_weights_of_layer(0.0, 0)
    for w in fcn.layers[0].weight.data:
        assert torch.equal(w, torch.tensor([0.0, 0.0, 0.0]))


def test_change_weights_with_wrong_type():
    fcn = _create_fcn()
    with pytest.raises(ValueError):
        fcn.set_weights_of_layer('4', 0)


def test_change_weights_with_wrong_dimension():
    fcn = _create_fcn()
    weights = np.array([[0], [3], [4], [5],
                        [6], [7], [8], [9]])
    with pytest.raises(ValueError):
        fcn.set_weights_of_layer(weights, 0)


def test_change_weights_of_layer_with_list():
    fcn = _create_fcn(in_dim={'x': 1})
    fcn.set_weights_of_layer([[0], [1], [2], [3], [4], [5],
                              [6], [7], [8], [9]], 0)
    i = 0.0
    for w in fcn.layers[0].weight.data:
        assert torch.equal(w, torch.tensor([i]))
        i = i + 1


def test_change_weights_of_layer_with_np_array():
    fcn = _create_fcn(in_dim={'x': 1})
    weights = np.array([[0], [1.0], [2], [3], [4], [5],
                        [6], [7], [8], [9]])
    fcn.set_weights_of_layer(weights, 0)
    i = 0
    for w in fcn.layers[0].weight.data:
        assert torch.equal(w, torch.DoubleTensor([i]))
        i = i + 1


def test_change_weights_of_layer_with_tensor():
    fcn = _create_fcn(in_dim={'x': 1})
    weights = torch.Tensor([[0], [1.0], [2], [3], [4], [5],
                            [6], [7], [8], [9]])
    fcn.set_weights_of_layer(weights, 0)
    i = 0.0
    for w in fcn.layers[0].weight.data:
        assert torch.equal(w, torch.tensor([i]))
        i = i + 1


def test_change_bias_of_layer_with_int():
    fcn = _create_fcn(in_dim={'x': 1})
    fcn.set_biases_of_layer(4, 0)
    assert torch.equal(fcn.layers[0].bias.data, 4*torch.ones(10))


def test_change_bias_with_wrong_dimension():
    fcn = _create_fcn()
    biases = np.array([[0], [1.0], [2], [3], [4], [5],
                       [6], [7], [8], [9]])
    with pytest.raises(ValueError):
        fcn.set_biases_of_layer(biases, 0)


def test_change_bias_of_layer_with_tensor():
    fcn = _create_fcn(in_dim={'x': 1})
    weights = torch.ones(10)
    fcn.set_biases_of_layer(weights, 0)
    for w in fcn.layers[0].bias.data:
        assert torch.equal(w, torch.tensor(1.0))


def test_change_activation_of_layer():
    fcn = _create_fcn(in_dim={'x': 1})
    fcn.set_activation_function_of_layer(torch.nn.ReLU(), 1)
    assert isinstance(fcn.layers[1], torch.nn.ReLU)


# Test BlockFCN:
def test_create_blockFCN():
    block = models.fcn.BlockFCN(variable_dims={'t':1}, 
                               solution_dims={'u': 2},
                               blocks=2, width=10)
    assert block.input_dim == 1
    assert block.width == 10
    assert block.blocks == 2
    assert block.output_dim == 2


def test_structur_of_blockFCN():
    block = models.fcn.BlockFCN(variable_dims={'t':1}, 
                               solution_dims={'u': 2},
                               blocks=2, width=10)
    assert isinstance(block.layers, torch.nn.ModuleList)
    for i in range(6):
        assert isinstance(block.layers[2*i], torch.nn.Linear)
    assert isinstance(block.layers[5], torch.nn.Tanh)
    assert isinstance(block.layers[9], torch.nn.Tanh)
    assert isinstance(block.layers[1], torch.nn.LeakyReLU)
    assert isinstance(block.layers[3], torch.nn.LeakyReLU)
    assert isinstance(block.layers[7], torch.nn.LeakyReLU)


def test_serialize_blockFCN():
    block = models.fcn.BlockFCN(variable_dims={'t':2}, 
                               solution_dims={'u': 2},
                               blocks=2, width=5)
    dic = block.serialize()
    assert dic['name'] == 'BlockFCN'
    assert dic['input_dim'] == 2
    assert dic['blocks'] == 2
    assert dic['width'] == 5
    assert dic['output_dim'] == 2


def test_forward_blockFCN():
    block = models.fcn.BlockFCN(variable_dims={'x': 2, 't': 1}, 
                               solution_dims={'u': 2},
                               blocks=2, width=10)
    input = {'x': torch.ones((2, 2)), 't': torch.zeros((2, 1))}
    output = block.forward(input)
    assert output['u'].shape == (2, 2)
    assert isinstance(output['u'], torch.Tensor)