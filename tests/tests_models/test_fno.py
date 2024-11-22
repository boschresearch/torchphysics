import torch
import pytest

from torchphysics.models.FNO import FNO, _FourierLayer
from torchphysics.problem.spaces import Points, R1, R2


def test_create_fourier_layer():
    fourier_layer = _FourierLayer(4, 4)
    assert fourier_layer.data_dim == 1
    assert fourier_layer.fourier_kernel.shape == (4, 4)


def test_create_fourier_layer_higher_dim():
    fourier_layer = _FourierLayer(8, (4, 6))
    assert fourier_layer.data_dim == 2
    assert fourier_layer.fourier_kernel.shape == (4, 6, 8)


def test_create_fourier_layer_with_linear_transform():
    fourier_layer = _FourierLayer(8, 4, linear_connection=True)
    assert isinstance(fourier_layer.linear_transform, torch.nn.Linear)
    assert fourier_layer.linear_transform.weight.shape == (8, 8)


def test_create_fourier_layer_with_batchnorm():
    fourier_layer = _FourierLayer(8, 4, space_res=10)
    assert fourier_layer.use_bn
    with pytest.raises(NotImplementedError):
        fourier_layer = _FourierLayer(8, (4, 4), space_res=10)


def test_forward_fourier_layer():
    fourier_layer = _FourierLayer(1, 10)
    input_data = torch.linspace(0, 1, 10).reshape(1, 10, 1)
    output_data = fourier_layer(input_data)
    assert output_data.shape[0] == 1
    assert output_data.shape[1] == 10
    assert output_data.shape[2] == 1


def test_forward_fourier_layer_with_multiple_transforms():
    fourier_layer = _FourierLayer(1, 10, linear_connection=True, 
                                  skip_connection=True, 
                                  space_res=10)
    input_data = torch.linspace(0, 1, 10).reshape(1, 10, 1)
    input_data = torch.repeat_interleave(input_data, 3, 0)
    output_data = fourier_layer(input_data)
    assert output_data.shape[0] == 3
    assert output_data.shape[1] == 10
    assert output_data.shape[2] == 1



def test_create_fno_default():
    fno = FNO(input_space=R2('f'), output_space=R1('u'), fourier_layers=4,
              activations=torch.nn.Tanh())
    assert isinstance(fno.fourier_sequential, torch.nn.Sequential)
    for i in range(0, 8, 2):
        assert isinstance(fno.fourier_sequential[i], _FourierLayer)
    for i in range(1, 8, 2):
        assert isinstance(fno.fourier_sequential[i], torch.nn.Tanh)
    assert isinstance(fno.channel_down_sampling, torch.nn.Module)
    assert isinstance(fno.channel_up_sampling, torch.nn.Module)


def test_create_fno_optional():
    in_network = torch.nn.Linear(2, 10)
    out_network = torch.nn.Linear(10, 1)
    fno = FNO(input_space=R2('f'), output_space=R1('u'), fourier_layers=2,
              fourier_modes=([3, 3], [4, 4]), linear_connections=False, hidden_channels=10,
              bias=[False, False],
              channel_up_sample_network=in_network,
              channel_down_sample_network=out_network,
              activations=torch.nn.Tanh())
    assert fno.channel_up_sampling == in_network
    assert fno.channel_down_sampling == out_network


def test_create_fno_set_modes():
    FNO(input_space=R2('f'), output_space=R1('u'), fourier_layers=4,
        fourier_modes=[3, 3])
    with pytest.raises(ValueError):
        FNO(input_space=R2('f'), output_space=R1('u'), fourier_layers=4,
            fourier_modes="a")
        

def test_forward_fno():
    fno = FNO(input_space=R1('f'), output_space=R2('u'), fourier_layers=4,
              activations=torch.nn.Tanh(), hidden_channels=10)
    input_data = torch.linspace(0, 1, 10).reshape(1, 10, 1)
    input_data = torch.repeat_interleave(input_data, 3, 0)
    output_data = fno(Points(input_data, R1("f"))).as_tensor
    assert output_data.shape[0] == 3
    assert output_data.shape[1] == 10
    assert output_data.shape[2] == 2