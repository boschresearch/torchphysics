import pytest
import collections as cln
import torch

from neural_diff_eq.models import SimpleFCN

input_dim = 2
input_dict = cln.OrderedDict()
input_dict['x'] = torch.randn([16, 1])
input_dict['t'] = torch.randn([16, 1])

def test_simplefcn():
    model = SimpleFCN(input_dim=input_dim)
    x = model(input_dict, track_gradients=['x'])
    assert input_dict['x'].requires_grad
    assert not input_dict['t'].requires_grad