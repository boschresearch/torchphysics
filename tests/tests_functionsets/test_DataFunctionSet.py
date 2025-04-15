import torch

from torchphysics.problem.spaces import R1, R2, FunctionSpace
from torchphysics.problem.domains.functionsets.data_functionset import DataFunctionSet

def test_create_data_fn_set():
    data = torch.rand((500, 100, 2))
    space = FunctionSpace(R1("x"), R2("u"))
    data_fn_set = DataFunctionSet(space, data)
    assert data_fn_set.function_set_size == 500


def test_data_fn_set_is_discrete():
    data = torch.rand((500, 100, 2))
    space = FunctionSpace(R1("x"), R2("u"))
    data_fn_set = DataFunctionSet(space, data)
    assert data_fn_set.is_discretized


def test_data_fn_set_create_fn():
    data = torch.rand((500, 100, 2))
    space = FunctionSpace(R1("x"), R2("u"))
    data_fn_set = DataFunctionSet(space, data)
    data_fn_set.create_functions()


def test_data_fn_set_get_fn():
    data = torch.rand((500, 100, 2))
    space = FunctionSpace(R1("x"), R2("u"))
    data_fn_set = DataFunctionSet(space, data)
    data_fn_set.create_functions()
    output_fn = data_fn_set.get_function(0)
    assert output_fn.space == R2("u")
    assert torch.all(output_fn.as_tensor == data[0])


def test_data_fn_set_get_fn_multiple():
    data = torch.rand((500, 100, 2))
    space = FunctionSpace(R1("x"), R2("u"))
    data_fn_set = DataFunctionSet(space, data)
    data_fn_set.create_functions()
    output_fn = data_fn_set.get_function([0, 101, 32])
    assert output_fn.space == R2("u")
    assert torch.all(output_fn.as_tensor[0] == data[0])
    assert torch.all(output_fn.as_tensor[1] == data[101])
    assert torch.all(output_fn.as_tensor[2] == data[32])


def test_data_fn_set_product():
    data = torch.rand((500, 100, 2))
    space = FunctionSpace(R1("x"), R2("u"))
    data_fn_set = DataFunctionSet(space, data)
    data2 = torch.rand((500, 100, 2))
    space2 = FunctionSpace(R1("x"), R2("w"))
    data_fn_set2 = DataFunctionSet(space2, data2)
    data_fn_set *= data_fn_set2
    data_fn_set.create_functions()
    output_fn = data_fn_set.get_function(0)
    assert output_fn.space == R2("u")*R2("w")
    assert output_fn.as_tensor.shape[-1] == 4


def test_data_fn_set_normalize():
    data = torch.rand((500, 100, 2))
    space = FunctionSpace(R1("x"), R2("u"))
    data_fn_set = DataFunctionSet(space, data)
    data_fn_set.compute_normalization()
    assert torch.is_tensor(data_fn_set.mean)
    assert torch.is_tensor(data_fn_set.std)


def test_data_fn_set_pca():
    data = torch.rand((500, 100, 2))
    space = FunctionSpace(R1("x"), R2("u"))
    data_fn_set = DataFunctionSet(space, data)
    data_fn_set.compute_pca(4)
    U, S, V = data_fn_set.pca
    assert torch.is_tensor(U)
    assert torch.is_tensor(S)
    assert torch.is_tensor(V)
    assert V.shape[0] == 200 and V.shape[1] == 4
    assert len(S) == 4
    data_fn_set.compute_pca(4, normalize_data=False)