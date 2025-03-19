import torch
import pytest
import math

from torchphysics.problem.spaces import R1, R2, FunctionSpace
from torchphysics.problem.domains.functionsets.grf_functionset import GRFFunctionSet


def test_create_grf_fn_set():
    X = R1("x")
    U = R1("u")
    fn_space = FunctionSpace(X, U)
    _ = GRFFunctionSet(fn_space, 100, 512)


def test_create_grf_fn_set_2D():
    X = R2("x")
    U = R1("u")
    fn_space = FunctionSpace(X, U)
    _ = GRFFunctionSet(fn_space, 100, (512, 512))


def test_create_grf_fn_set_wrong_dimension():
    X = R2("x")
    U = R1("u")
    fn_space = FunctionSpace(X, U)
    with pytest.raises(AssertionError):
        _ = GRFFunctionSet(fn_space, 100, 512)


def test_grf_is_discrete():
    X = R2("x")
    U = R1("u")
    fn_space = FunctionSpace(X, U)
    grf_fn_set = GRFFunctionSet(fn_space, 100, (512, 512))
    assert grf_fn_set.is_discretized


def test_create_grf():
    X = R1("x")
    U = R1("u")
    fn_space = FunctionSpace(X, U)
    grf_fn_set = GRFFunctionSet(fn_space, 100, 512)
    grf_fn_set.create_functions()
    assert torch.is_tensor(grf_fn_set.grf)
    assert grf_fn_set.grf.shape[0] == 100
    assert grf_fn_set.grf.shape[1] == 512
    assert grf_fn_set.grf.shape[2] == 1


def test_create_grf_no_normalize():
    X = R1("x")
    U = R1("u")
    fn_space = FunctionSpace(X, U)
    grf_fn_set = GRFFunctionSet(fn_space, 100, 16, 
                                normalize=False, sample_noise_in_fourier_space=False)
    grf_fn_set.create_functions()
    assert grf_fn_set.grf.shape[0] == 100
    assert grf_fn_set.grf.shape[1] == 16
    assert grf_fn_set.grf.shape[2] == 1


def test_create_grf_2D():
    X = R2("x")
    U = R1("u")
    fn_space = FunctionSpace(X, U)
    grf_fn_set = GRFFunctionSet(fn_space, 100, (64, 64), flatten=False)
    grf_fn_set.create_functions()
    assert grf_fn_set.grf.shape[0] == 100
    assert grf_fn_set.grf.shape[1] == 64
    assert grf_fn_set.grf.shape[2] == 64
    assert grf_fn_set.grf.shape[3] == 1


def test_create_grf_2D_flatten():
    X = R2("x")
    U = R1("u")
    fn_space = FunctionSpace(X, U)
    grf_fn_set = GRFFunctionSet(fn_space, 100, (64, 64), flatten=True)
    grf_fn_set.create_functions()
    assert grf_fn_set.grf.shape[0] == 100
    assert grf_fn_set.grf.shape[1] == 64*64
    assert grf_fn_set.grf.shape[2] == 1


def test_create_grf_custom_conv_fn():
    X = R2("x")
    U = R1("u")
    fn_space = FunctionSpace(X, U)
    def conv_fn(x):
        return 1/(1.0 + 0.9*math.sqrt(abs(x[0])) + 0.9*abs(x[1])**2)**2
    grf_fn_set = GRFFunctionSet(fn_space, 100, (64, 64), flatten=False, auto_cov_fn=conv_fn)
    grf_fn_set.create_functions()
    assert grf_fn_set.grf.shape[0] == 100
    assert grf_fn_set.grf.shape[1] == 64
    assert grf_fn_set.grf.shape[2] == 64
    assert grf_fn_set.grf.shape[3] == 1


def test_get_grf():
    X = R1("x")
    U = R1("u")
    fn_space = FunctionSpace(X, U)
    grf_fn_set = GRFFunctionSet(fn_space, 100, 512)
    grf_fn_set.create_functions()
    random_field = grf_fn_set.get_function([0, 12]).as_tensor
    assert torch.is_tensor(random_field)
    assert random_field.shape[0] == 2
    assert random_field.shape[1] == 512
    assert random_field.shape[2] == 1