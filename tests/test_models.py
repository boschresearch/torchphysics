import torch
import pytest
import neural_diff_eq.models as models


def test_prepare_input():
    Dmodel = models.fcn.DiffEqModel()
    input = {'x': torch.ones((2, 2)), 't': torch.zeros((2, 1))}
    out = Dmodel._prepare_inputs(input)
    assert torch.equal(out, torch.FloatTensor([[1, 1, 0], [1, 1, 0]]))
    input = {'x': torch.ones((2, 2))}
    out = Dmodel._prepare_inputs(input)
    assert torch.equal(out, torch.FloatTensor([[1, 1], [1, 1]]))


def test_serialize_diffeqmodel():
    Dmodel = models.DiffEqModel()
    with pytest.raises(NotImplementedError):
        Dmodel.serialize()


# Test SimpleFCN:
def test_create_simpleFCN():
    fcn = models.fcn.SimpleFCN(input_dim=1, width=10, depth=2, output_dim=2)
    assert fcn.input_dim == 1
    assert fcn.width == 10
    assert fcn.depth == 2
    assert fcn.output_dim == 2


def test_structur_of_simpleFCN():
    fcn = models.fcn.SimpleFCN(input_dim=2, width=10, depth=3, output_dim=1)
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
    fcn = models.fcn.SimpleFCN(input_dim=2)
    dic = fcn.serialize()
    assert dic['name'] == 'SimpleFCN'
    assert dic['input_dim'] == 2
    assert dic['depth'] == 3
    assert dic['width'] == 20
    assert dic['output_dim'] == 1


def test_forward_simpleFCN():
    fcn = models.fcn.SimpleFCN(input_dim=3)
    input = {'x': torch.ones((2, 2)), 't': torch.zeros((2, 1))}
    output = fcn.forward(input)
    assert output.shape == (2, 1)
    assert isinstance(output, torch.Tensor)


# Test SimpleFCN:
def test_create_blockFCN():
    block = models.fcn.BlockFCN(input_dim=1, width=10, blocks=2, output_dim=2)
    assert block.input_dim == 1
    assert block.width == 10
    assert block.blocks == 2
    assert block.output_dim == 2


def test_structur_of_blockFCN():
    block = models.fcn.BlockFCN(input_dim=1, width=10, blocks=2, output_dim=2)
    assert isinstance(block.layers, torch.nn.ModuleList)
    for i in range(6):
        assert isinstance(block.layers[2*i], torch.nn.Linear)
    assert isinstance(block.layers[5], torch.nn.Tanh)
    assert isinstance(block.layers[9], torch.nn.Tanh)
    assert isinstance(block.layers[1], torch.nn.LeakyReLU)
    assert isinstance(block.layers[3], torch.nn.LeakyReLU)
    assert isinstance(block.layers[7], torch.nn.LeakyReLU)


def test_serialize_blockFCN():
    block = models.fcn.BlockFCN(input_dim=2, width=5, blocks=2, output_dim=2)
    dic = block.serialize()
    assert dic['name'] == 'BlockFCN'
    assert dic['input_dim'] == 2
    assert dic['blocks'] == 2
    assert dic['width'] == 5
    assert dic['output_dim'] == 2


def test_forward_blockFCN():
    block = models.fcn.BlockFCN(input_dim=3, width=5, blocks=2, output_dim=2)
    input = {'x': torch.ones((2, 2)), 't': torch.zeros((2, 1))}
    output = block.forward(input)
    assert output.shape == (2, 2)
    assert isinstance(output, torch.Tensor)