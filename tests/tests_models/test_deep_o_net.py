import torch
import pytest

from torchphysics.models.deeponet.branchnets import (BranchNet, FCBranchNet, ConvBranchNet)
from torchphysics.models.deeponet.trunknets import (TrunkNet, FCTrunkNet) 
from torchphysics.models.deeponet.deeponet import DeepONet
from torchphysics.models.deeponet.layers import TrunkLinear
from torchphysics.models.model import Sequential, NormalizationLayer
from torchphysics.problem.spaces import Points, R1, R2, FunctionSpace
from torchphysics.problem.domains import Interval, CustomFunctionSet
from torchphysics.problem.samplers.grid_samplers import GridSampler

"""
Tests for trunk net:
"""

def test_create_trunk_net():
    default_grid = torch.rand((1, 100, 2))
    net = TrunkNet(input_space=R2('x'), default_trunk_input=default_grid)
    assert net.input_space == R2('x')
    assert net.output_space == None
    assert net.output_neurons == 0


def test_create_fc_trunk_net():
    default_grid = torch.rand((1, 100, 2))
    net = FCTrunkNet(input_space=R2('x'), default_trunk_input=default_grid)
    assert net.input_space == R2('x')
    assert net.output_space == None
    assert net.output_neurons == 0

"""
Tests for branch net:
"""
def helper_fn_set():
    def f(k, t):
        return k*t
    params = Interval(R1('k'), 0, 1)
    fn_space = FunctionSpace(R1('t'), R1('e'))
    fn_set =  CustomFunctionSet(fn_space, GridSampler(params, 20), f)
    return fn_space, fn_set


def test_create_branch_net():
    fn_space, _ = helper_fn_set()
    net = BranchNet(fn_space, grid=torch.rand((1, 10, 1)))
    assert net.input_space == fn_space


def test_create_fc_branch_net():
    fn_space, _ = helper_fn_set()
    net = FCBranchNet(fn_space, grid=torch.rand((1, 15, 1)))
    assert net.input_space == fn_space


def test_fix_branch_net_with_function_set():
    fn_space, fn_set = helper_fn_set()
    net = FCBranchNet(fn_space, grid=torch.rand((1, 15, 1)))
    net.finalize(R1('u'), 20)
    net.fix_input(fn_set)


def test_fix_branch_wrong_input():
    fn_space, _ = helper_fn_set()
    net = FCBranchNet(fn_space, grid=torch.rand((1, 15, 1)))
    with pytest.raises(NotImplementedError):
        net.fix_input(34)

"""
Tests for DeepONet:
"""
def test_create_deeponet():
    default_grid = torch.rand((1, 100, 1))
    trunk = TrunkNet(input_space=R1('t'), default_trunk_input=default_grid)
    fn_space, _ = helper_fn_set()
    branch = FCBranchNet(fn_space, grid=default_grid)
    net = DeepONet(trunk, branch, output_space=R1('u'), output_neurons=20)
    assert net.trunk == trunk
    assert net.branch == branch
    assert net.input_space == R1('t')
    assert net.output_space == R1('u')
    assert net.trunk.output_space == R1('u')
    assert net.branch.output_space == R1('u')
    assert net.trunk.output_neurons == 20
    assert net.branch.output_neurons == 20


def test_create_deeponet_with_seq_trunk():
    default_grid = torch.rand((1, 100, 1))
    trunk = TrunkNet(input_space=R1('t'), default_trunk_input=default_grid)
    fn_space, _ = helper_fn_set()
    branch = FCBranchNet(fn_space, grid=default_grid)
    seq_trunk = Sequential(NormalizationLayer(Interval(R1('t'), 0, 1)), trunk)
    net = DeepONet(seq_trunk, branch, output_space=R1('u'), output_neurons=20)
    assert net.trunk == seq_trunk
    assert net.branch == branch
    assert net.input_space == R1('t')
    assert net.output_space == R1('u')
    

def test_deeponet_fix_branch():
    def f(t):
        return 20*t
    default_grid = torch.rand((1, 100, 1))
    trunk = TrunkNet(input_space=R1('t'), default_trunk_input=default_grid)
    fn_space, _ = helper_fn_set()
    branch = FCBranchNet(fn_space, grid=default_grid)
    net = DeepONet(trunk, branch, output_space=R1('u'), output_neurons=20)
    net.fix_branch_input(f)
    assert branch.current_out.shape == (1, 1, 20)


def test_deeponet_forward():
    def f(t):
        return 20*t
    default_grid = torch.rand((1, 100, 1))
    trunk = FCTrunkNet(input_space=R1('t'), default_trunk_input=default_grid)
    fn_space, _ = helper_fn_set()
    branch = FCBranchNet(fn_space, grid=default_grid)
    net = DeepONet(trunk, branch, output_space=R1('u'), output_neurons=20)
    test_data = Points(torch.tensor([[[2], [0], [3.4], [2.9]]]), R1('t'))
    out = net(test_data, f)
    assert 'u' in out.space
    assert out.as_tensor.shape == (1, 4, 1)


def test_deeponet_forward_multi_dim_output():
    def f(t):
        return 20*t
    default_grid = torch.rand((1, 100, 1))
    trunk = FCTrunkNet(input_space=R1('t'), default_trunk_input=default_grid)
    fn_space, _ = helper_fn_set()
    branch = FCBranchNet(fn_space, grid=default_grid)
    net = DeepONet(trunk, branch, output_space=R2('u'), output_neurons=20)
    test_data = Points(torch.tensor([[[2], [0], [3.4], [2.9]]]), R1('t'))
    out = net(test_data, f)
    assert 'u' in out.space
    assert out.as_tensor.shape == (1, 4, 2)


def test_deeponet_forward_with_fixed_branch():
    def f(t):
        return torch.sin(t)
    default_grid = torch.rand((1, 100, 1))
    trunk = FCTrunkNet(input_space=R1('t'), default_trunk_input=default_grid)
    fn_space, _ = helper_fn_set()
    branch = FCBranchNet(fn_space, grid=default_grid)
    net = DeepONet(trunk, branch, output_space=R1('u'), output_neurons=20)
    test_data = Points(torch.tensor([[[2], [0], [3.4], [2.9], [5.2]]]), R1('t'))
    net.fix_branch_input(f)
    out = net(test_data)
    assert 'u' in out.space
    assert out.as_tensor.shape == (1, 5, 1)


def test_deeponet_forward_branch_intern():
    default_grid = torch.rand((1, 100, 1))
    trunk = FCTrunkNet(input_space=R1('t'), default_trunk_input=default_grid)
    fn_space, fn_set = helper_fn_set()
    branch = FCBranchNet(fn_space, grid=default_grid)
    net = DeepONet(trunk, branch, output_space=R1('u'), output_neurons=20)
    net._forward_branch(fn_set, iteration_num=0)


def test_trunk_linear():
    linear_a = TrunkLinear(30, 20, bias=True)
    linear_b = torch.nn.Linear(30, 20, bias=True)
    linear_b.bias = torch.nn.Parameter(linear_a.bias[:])
    linear_b.weight = torch.nn.Parameter(linear_a.weight[:])

    x = torch.randn(2,4,30).expand(5,-1,-1,-1)
    x.requires_grad = True
    
    y_b = linear_b(x)
    y_a = linear_a(x)
    
    assert torch.norm(
        torch.autograd.grad(y_a.sum(), x, retain_graph=True)[0]-torch.autograd.grad(y_b.sum(), x,retain_graph=True)[0]
    ) < 1e-5

    y_a.sum().backward()
    y_b.sum().backward()
    assert torch.norm(linear_a.weight.grad - linear_b.weight.grad) < 1e-2
    assert torch.norm(linear_a.bias.grad - linear_b.bias.grad) < 1e-5