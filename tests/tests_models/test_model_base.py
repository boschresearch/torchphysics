import pytest
import torch

from torchphysics.models import *
from torchphysics.problem.spaces import R1, R2, Points
from torchphysics.problem.domains import Interval, Circle
from torchphysics.problem.samplers import RandomUniformSampler
from torchphysics.problem.conditions.condition import SquaredError


def test_base_model_creation():
    in_space = R2('x')*R1('t')
    out_space = R2('u')
    model = Model(input_space=in_space, output_space=out_space)
    assert isinstance(model, torch.nn.Module)
    assert model.input_space == in_space
    assert model.output_space == out_space


def test_create_normalization_layer():
    in_space = R2('x')
    C = Circle(in_space, [1, 0], 2.0)
    model = NormalizationLayer(C)
    assert isinstance(model, torch.nn.Module)
    assert torch.equal(model.normalize.bias, torch.tensor([-0.5, 0.0]))
    assert torch.equal(model.normalize.weight, torch.tensor([[0.5, 0.0],
                                                             [0.0, 0.5]]))
    assert model.input_space == in_space
    assert model.output_space == in_space


def test_create_normalization_layer_for_product_domain():
    in_space = R2('x')*R1('t')
    C = Circle(R2('x'), [1, 0], 2.0) * Interval(R1('t'), 0, 1)
    model = NormalizationLayer(C)
    assert isinstance(model, torch.nn.Module)
    assert torch.equal(model.normalize.bias, torch.tensor([-0.5, 0.0, -1.0]))
    assert torch.equal(model.normalize.weight, torch.tensor([[0.5, 0.0, 0.0],
                                                             [0.0, 0.5, 0.0], 
                                                             [0.0, 0.0, 2.0]]))
    assert model.input_space == in_space
    assert model.output_space == in_space


def test_normalization_layer_forward():
    in_space = R2('x')
    C = Circle(in_space, [1, 0], 2.0)
    model = NormalizationLayer(C)
    ps = RandomUniformSampler(C, n_points=50)
    out = model(ps.sample_points())
    assert isinstance(out, Points)
    assert out.as_tensor.shape == (50, 2)
    assert torch.all(torch.linalg.norm(out, dim=1) <=1)


def test_create_adaptive_weight_layer():
    model = AdaptiveWeightLayer(10)
    assert isinstance(model, torch.nn.Module)
    assert model.weight.shape == torch.Size([10])


def test_adaptive_weight_forward():
    model = AdaptiveWeightLayer(2)
    inp_points = torch.tensor([2.0, 1.0])
    out = model(inp_points)
    assert isinstance(out, torch.Tensor)
    assert out.shape == torch.Size([2])


def test_sequential_with_normalization():
    out_space = R1('u')
    fcn = FCN(input_space=R2('x'), output_space=out_space, 
              hidden=(10, ))
    in_space = R2('x')
    C = Circle(in_space, [1, 0], 2.0)
    normalize_model = NormalizationLayer(C)
    seq = Sequential(normalize_model, fcn)
    ps = RandomUniformSampler(C, n_points=50)
    out = seq(ps.sample_points())
    assert seq.input_space == in_space
    assert seq.output_space == out_space
    assert isinstance(out, Points)
    assert out.as_tensor.shape == (50, 1)
    assert out.requires_grad


def test_sequential_with_weights():
    out_space = R2('u')
    in_space = R2('x')
    fcn = FCN(input_space=in_space, output_space=out_space, 
              hidden=(10, ))
    C = Circle(in_space, [1, 0], 2.0)
    adap_model = AdaptiveWeightLayer(50)
    ps = RandomUniformSampler(C, n_points=50)
    out = fcn(ps.sample_points())
    assert out.space == out_space
    sq = SquaredError()
    out = sq(out)
    out = adap_model(out)
    assert out.shape == torch.Size([50])
    assert out.requires_grad


def test_create_parallel_model():
    fcn1 = FCN(input_space=R2('x'), output_space=R1('u'), 
               hidden=(10, ))
    fcn2 = FCN(input_space=R2('x'), output_space=R1('v'), 
               hidden=(10, ))
    parallel = Parallel(fcn1, fcn2)
    assert parallel.input_space == R2('x')
    assert parallel.output_space == R1('u')*R1('v')


def test_create_parallel_model_different_input_spaces():
    fcn1 = FCN(input_space=R2('x'), output_space=R1('u'), 
               hidden=(10, ))
    fcn2 = FCN(input_space=R1('t'), output_space=R1('v'), 
               hidden=(10, ))
    parallel = Parallel(fcn1, fcn2)
    assert parallel.input_space == R2('x')*R1('t')
    assert parallel.output_space == R1('u')*R1('v')


def test_cant_create_parallel_model_with_same_output():
    fcn1 = FCN(input_space=R2('x'), output_space=R1('u'), 
               hidden=(10, ))
    fcn2 = FCN(input_space=R2('x'), output_space=R1('u'), 
               hidden=(10, ))
    with pytest.raises(AssertionError):
        Parallel(fcn1, fcn2)


def test_parallel_model_forward():
    fcn1 = FCN(input_space=R2('x'), output_space=R1('u'), 
               hidden=(10, ))
    fcn2 = FCN(input_space=R2('x'), output_space=R2('v'), 
               hidden=(10, ))
    parallel = Parallel(fcn1, fcn2)
    inp = Points(torch.tensor([[0.0, 0.0], [2.0, 2.0], [2.0, 3.0]]), R2('x'))
    out = parallel(inp)
    assert isinstance(out, Points)
    assert out.space == R1('u')*R2('v')
    assert out.as_tensor.shape == (3, 3)


def test_parallel_model_forward_different_input_spaces():
    fcn1 = FCN(input_space=R2('x'), output_space=R1('u'), 
               hidden=(10, ))
    fcn2 = FCN(input_space=R1('t'), output_space=R1('v'), 
               hidden=(10, ))
    parallel = Parallel(fcn1, fcn2)
    inp = Points(torch.tensor([[0.0, 0.0, 1.0], [2.0, 2.0, 0.0]]), R2('x')*R1('t'))
    out = parallel(inp)
    assert isinstance(out, Points)
    assert out.space == R1('u')*R1('v')
    assert out.as_tensor.shape == (2, 2)