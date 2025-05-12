import torch
import pytest

from torchphysics.models.PCANN import PCANN_FC, PCANN
from torchphysics.problem.spaces import R1, R2, FunctionSpace
from torchphysics.problem.domains.functionsets.data_functionset import DataFunctionSet

def test_pcann_create():
    T = R2("t")
    U = R1("u") 
    model = PCANN_FC(T, U, 
        (torch.rand(100, 4), torch.rand(4), torch.rand(20, 4)), 
        (torch.rand(100, 6), torch.rand(6), torch.rand(30, 6)),
        output_shape=[30]
    )
    assert torch.is_tensor(model.mean_in)
    assert torch.is_tensor(model.mean_out)
    assert torch.is_tensor(model.std_in)
    assert torch.is_tensor(model.std_out)
    assert torch.is_tensor(model.eigenvalues_out)
    assert torch.is_tensor(model.eigenvalues_in)
    assert torch.is_tensor(model.eigenvectors_out)
    assert torch.is_tensor(model.eigenvectors_in)
    assert len(model.eigenvalues_in) == 4
    assert len(model.eigenvalues_out) == 6


def test_pcann_create_from_fn_set():
    T = R2("t")
    U = R1("u") 
    data = torch.rand((500, 100, 2))
    space = FunctionSpace(U, T)
    data_fn_set = DataFunctionSet(space, data)
    data_fn_set.compute_pca(12)
    model = PCANN_FC.from_fn_set(
        data_fn_set, data_fn_set
    )
    assert torch.is_tensor(model.mean_in)
    assert torch.is_tensor(model.mean_out)
    assert torch.is_tensor(model.std_in)
    assert torch.is_tensor(model.std_out)
    assert torch.is_tensor(model.eigenvalues_out)
    assert torch.is_tensor(model.eigenvalues_in)
    assert torch.is_tensor(model.eigenvectors_out)
    assert torch.is_tensor(model.eigenvectors_in)
    assert len(model.eigenvalues_in) == 12
    assert len(model.eigenvalues_out) == 12


def test_pcann_apply_network_gives_error():
    T = R2("t")
    U = R1("u") 
    model = PCANN(T, U, 
        (torch.rand(100, 4), torch.rand(4), torch.rand(20, 4)), 
        (torch.rand(100, 6), torch.rand(6), torch.rand(30, 6)),
        output_shape=[30]
    )
    with pytest.raises(NotImplementedError):
        _ = model.apply_network(None)


def test_pcann_forward():
    T = R2("t")
    U = R1("u") 
    space = FunctionSpace(U, T)
    data = torch.rand((500, 100, 2))
    data_out = torch.rand((500, 50, 50, 2))
    data_fn_set = DataFunctionSet(space, data)
    data_fn_set_out = DataFunctionSet(space, data_out)
    data_fn_set.compute_pca(12)
    data_fn_set_out.compute_pca(20)
    model = PCANN_FC.from_fn_set(data_fn_set, data_fn_set_out)
    
    input_data = torch.rand((30, 100, 2))
    output_data = model(input_data).as_tensor
    assert torch.is_tensor(output_data)
    assert output_data.shape[0] == 30
    assert output_data.shape[1] == 50 and output_data.shape[2] == 50
    assert output_data.shape[3] == 2