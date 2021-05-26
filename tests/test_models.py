import pytest
import collections as cln
import torch

from neural_diff_eq.models import SimpleFCN

input_dim = 2


def _input_dict():
    input_dict = cln.OrderedDict()
    input_dict['x'] = torch.randn([16, 1])
    input_dict['t'] = torch.randn([16, 1])
    return input_dict


def test_grad_tracking():
    model = SimpleFCN(input_dim=input_dim)

    x1 = _input_dict()
    _ = model(x1, track_gradients=['x'])
    assert x1['x'].requires_grad
    assert not x1['t'].requires_grad

    x2 = _input_dict()
    _ = model(x2, track_gradients=True)
    assert x2['x'].requires_grad
    assert x2['t'].requires_grad

    x3 = _input_dict()
    _ = model(x3, track_gradients=False)
    assert not x3['x'].requires_grad
    assert not x3['t'].requires_grad
