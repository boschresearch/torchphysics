import torch

from torchphysics.problem.spaces import R1, R2, R3, FunctionSpace, Points
from torchphysics.problem.domains.functionsets.harmonic_functionset import (
    HarmonicFunctionSet1D, HarmonicFunctionSet2D, HarmonicFunctionSet3D
)

def test_create_harmonic_fn_set_1D():
    T = R1("t")
    U = R1("u")
    Fn_space = FunctionSpace(T, U)
    harmonic_fns = HarmonicFunctionSet1D(Fn_space, 1000, 3, 5)
    assert harmonic_fns.period_len == 3
    assert harmonic_fns.max_frequence == 5


def test_harmonic_fn_set_1D_build_functions():
    T = R1("t")
    U = R1("u")
    Fn_space = FunctionSpace(T, U)
    harmonic_fns = HarmonicFunctionSet1D(Fn_space, 1000, 3, 5) 
    harmonic_fns.create_functions()
    assert torch.is_tensor(harmonic_fns.fourier_coefficients)
    assert harmonic_fns.fourier_coefficients.shape[0] == 1000
    assert harmonic_fns.fourier_coefficients.shape[1] == 6
    assert harmonic_fns.fourier_coefficients.shape[2] == 2


def test_harmonic_fn_set_1D_get_functions():
    T = R1("t")
    U = R1("u")
    Fn_space = FunctionSpace(T, U)
    harmonic_fns = HarmonicFunctionSet1D(Fn_space, 1000, 3, 5) 
    harmonic_fns.create_functions()
    fn = harmonic_fns.get_function([1, 2, 3, 4, 5])
    assert callable(fn)
    fn_out = fn(Points(torch.rand((1, 100, 1)), T)).as_tensor
    assert torch.is_tensor(fn_out)
    assert fn_out.shape[0] == 5
    assert fn_out.shape[1] == 100
    assert fn_out.shape[2] == 1

    fn_out = fn(Points(torch.rand((5, 100, 1)), T)).as_tensor
    assert torch.is_tensor(fn_out)
    assert fn_out.shape[0] == 5
    assert fn_out.shape[1] == 100
    assert fn_out.shape[2] == 1


def test_create_harmonic_fn_set_2D():
    T = R2("t")
    U = R1("u")
    Fn_space = FunctionSpace(T, U)
    harmonic_fns = HarmonicFunctionSet2D(Fn_space, 1000, (3, 3), (2, 2))
    assert harmonic_fns.period_len == (3, 3)
    assert harmonic_fns.max_frequence == (2, 2)


def test_harmonic_fn_set_2D_build_functions():
    T = R2("t")
    U = R1("u")
    Fn_space = FunctionSpace(T, U)
    harmonic_fns = HarmonicFunctionSet2D(Fn_space, 1000, (3, 3), (6, 2)) 
    harmonic_fns.create_functions()
    assert torch.is_tensor(harmonic_fns.fourier_coefficients)
    assert harmonic_fns.fourier_coefficients.shape[0] == 1000
    assert harmonic_fns.fourier_coefficients.shape[1] == 7
    assert harmonic_fns.fourier_coefficients.shape[2] == 3
    assert harmonic_fns.fourier_coefficients.shape[3] == 4


def test_harmonic_fn_set_2D_get_functions():
    T = R2("t")
    U = R1("u")
    Fn_space = FunctionSpace(T, U)
    harmonic_fns = HarmonicFunctionSet2D(Fn_space, 1000, (3, 3), (2, 5)) 
    harmonic_fns.create_functions()
    fn = harmonic_fns.get_function([1, 2, 3, 4, 5])
    assert callable(fn)
    fn_out = fn(Points(torch.rand((1, 100, 2)), T)).as_tensor
    assert torch.is_tensor(fn_out)
    assert fn_out.shape[0] == 5
    assert fn_out.shape[1] == 100
    assert fn_out.shape[2] == 1

    fn_out = fn(Points(torch.rand((5, 100, 2)), T)).as_tensor
    assert torch.is_tensor(fn_out)
    assert fn_out.shape[0] == 5
    assert fn_out.shape[1] == 100
    assert fn_out.shape[2] == 1



def test_create_harmonic_fn_set_3D():
    T = R3("t")
    U = R1("u")
    Fn_space = FunctionSpace(T, U)
    harmonic_fns = HarmonicFunctionSet3D(Fn_space, 1000, (3, 3, 3), (6, 2, 1)) 
    assert harmonic_fns.period_len == (3, 3, 3)
    assert harmonic_fns.max_frequence == (6, 2, 1)


def test_harmonic_fn_set_3D_build_functions():
    T = R3("t")
    U = R1("u")
    Fn_space = FunctionSpace(T, U)
    harmonic_fns = HarmonicFunctionSet3D(Fn_space, 1000, (3, 3, 3), (6, 2, 1)) 
    harmonic_fns.create_functions()
    assert torch.is_tensor(harmonic_fns.fourier_coefficients)
    assert harmonic_fns.fourier_coefficients.shape[0] == 1000
    assert harmonic_fns.fourier_coefficients.shape[1] == 7
    assert harmonic_fns.fourier_coefficients.shape[2] == 3
    assert harmonic_fns.fourier_coefficients.shape[3] == 2
    assert harmonic_fns.fourier_coefficients.shape[4] == 8


def test_harmonic_fn_set_3D_get_functions():
    T = R3("t")
    U = R1("u")
    Fn_space = FunctionSpace(T, U)
    harmonic_fns = HarmonicFunctionSet3D(Fn_space, 1000, (3, 3, 3), (2, 5, 3)) 
    harmonic_fns.create_functions()
    fn = harmonic_fns.get_function([1, 2, 3, 4, 5])
    assert callable(fn)
    fn_out = fn(Points(torch.rand((1, 100, 3)), T)).as_tensor
    assert torch.is_tensor(fn_out)
    assert fn_out.shape[0] == 5
    assert fn_out.shape[1] == 100
    assert fn_out.shape[2] == 1

    fn_out = fn(Points(torch.rand((5, 100, 3)), T)).as_tensor
    assert torch.is_tensor(fn_out)
    assert fn_out.shape[0] == 5
    assert fn_out.shape[1] == 100
    assert fn_out.shape[2] == 1