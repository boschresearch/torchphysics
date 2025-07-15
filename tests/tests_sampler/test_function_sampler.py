import torch
import pytest

from torchphysics.problem.spaces import R1, FunctionSpace
from torchphysics.problem.domains.functionsets import CustomFunctionSet
from torchphysics.problem.samplers import GridSampler
from torchphysics.problem.domains.domain1D import Interval
from torchphysics.problem.samplers.function_sampler import (
    FunctionSampler, FunctionSamplerRandomUniform, FunctionSamplerOrdered, FunctionSamplerCoupled
)


def make_default_fn_set():
    sampler = GridSampler(Interval(R1("k"), 0, 1), 100)
    custom_fn = lambda k : k+2
    space = FunctionSpace(R1("x"), R1("u"))
    return CustomFunctionSet(space, sampler, custom_fn)


def test_create_function_sampler():
    fn_set = make_default_fn_set()
    fn_sampler = FunctionSampler(20, fn_set, 100)
    assert len(fn_sampler.current_indices) == 20
    assert fn_sampler.function_creation_interval == 100
    assert fn_sampler.n_functions == 20
    assert fn_sampler.iteration_counter == 100


def test_function_sampler_recreation_check():
    fn_set = make_default_fn_set()
    fn_sampler = FunctionSampler(20, fn_set, 100)
    fn_sampler._check_recreate_functions()
    assert fn_sampler.iteration_counter == 0


def test_function_sampler_recreation_check_two_times():
    fn_set = make_default_fn_set()
    fn_sampler = FunctionSampler(20, fn_set, 100)
    fn_sampler._check_recreate_functions()
    fn_sampler._check_recreate_functions()
    assert fn_sampler.iteration_counter == 1


def test_create_random_uniform_function_sampler():
    fn_set = make_default_fn_set()
    fn_sampler = FunctionSamplerRandomUniform(20, fn_set, 100)
    assert len(fn_sampler.current_indices) == 20
    assert fn_sampler.function_creation_interval == 100
    assert fn_sampler.n_functions == 20
    assert fn_sampler.iteration_counter == 100


def test_random_uniform_function_sampler_sample():
    fn_set = make_default_fn_set()
    fn_sampler = FunctionSamplerRandomUniform(20, fn_set, 100)
    fns = fn_sampler.sample_functions()
    assert torch.all(fn_sampler.current_indices < 100)
    assert callable(fns)


def test_create_ordered_function_sampler():
    fn_set = make_default_fn_set()
    fn_sampler = FunctionSamplerOrdered(20, fn_set, 100)
    assert len(fn_sampler.current_indices) == 20
    assert fn_sampler.function_creation_interval == 100
    assert fn_sampler.n_functions == 20
    assert fn_sampler.iteration_counter == 100


def test_ordered_function_sampler_sample():
    fn_set = make_default_fn_set()
    fn_sampler = FunctionSamplerOrdered(20, fn_set, 100)
    fns = fn_sampler.sample_functions()
    assert torch.all(fn_sampler.new_indieces >= 20)
    assert torch.all(fn_sampler.new_indieces < 40)
    assert callable(fns)


def test_ordered_function_sampler_sample_two_times():
    fn_set = make_default_fn_set()
    fn_sampler = FunctionSamplerOrdered(20, fn_set, 100)
    fns = fn_sampler.sample_functions()
    fns = fn_sampler.sample_functions()
    assert torch.all(fn_sampler.new_indieces >= 40)
    assert torch.all(fn_sampler.new_indieces < 60)
    assert callable(fns)


def test_ordered_function_sampler_sample_multiple_times():
    fn_set = make_default_fn_set()
    fn_sampler = FunctionSamplerOrdered(20, fn_set, 100)
    for _ in range(5):
        _ = fn_sampler.sample_functions()
    assert torch.all(fn_sampler.new_indieces >= 0)
    assert torch.all(fn_sampler.new_indieces < 20)


def test_create_coupled_function_sampler():
    fn_set = make_default_fn_set()
    fn_sampler = FunctionSamplerOrdered(20, fn_set, 100)
    fn_sampler2 = FunctionSamplerCoupled(fn_set, fn_sampler)
    assert fn_sampler2.function_creation_interval == 100
    assert fn_sampler2.n_functions == 20
    assert fn_sampler2.iteration_counter == 100


def test_coupled_function_sampler_sample():
    fn_set = make_default_fn_set()
    fn_sampler = FunctionSamplerOrdered(20, fn_set, 100)
    fn_sampler2 = FunctionSamplerCoupled(fn_set, fn_sampler)
    _ = fn_sampler.sample_functions()
    _ = fn_sampler2.sample_functions()
    assert torch.all(fn_sampler2.current_indices >= 0)
    assert torch.all(fn_sampler2.current_indices < 20)