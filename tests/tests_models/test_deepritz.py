import torch

from torchphysics.models.deepritz import DeepRitzNet
from torchphysics.problem.spaces import Points, R1, R2


def test_create_deepritz_model():
    net = DeepRitzNet(input_space=R2('x'), output_space=R1('u'),
                      width=10, depth=3)
    assert isinstance(net.linearIn, torch.nn.Linear)
    assert net.linearIn.in_features == 2
    assert net.linearIn.out_features == 10
    for i in range(3):
        assert isinstance(net.linear1[i], torch.nn.Linear)
        assert net.linear1[i].in_features == 10
        assert net.linear1[i].out_features == 10
    for i in range(3):
        assert isinstance(net.linear2[i], torch.nn.Linear)
        assert net.linear2[i].in_features == 10
        assert net.linear2[i].out_features == 10
    assert isinstance(net.linearOut, torch.nn.Linear)
    assert net.linearOut.in_features == 10
    assert net.linearOut.out_features == 1


def test_forward():
    net = DeepRitzNet(input_space=R2('x'), output_space=R2('u'),
                      width=15, depth=3)
    test_data = Points(torch.tensor([[2, 3.0], [0, 1]]), R2('x'))
    out = net(test_data)
    assert isinstance(out, Points)
    assert out._t.size() == torch.Size([2, 2])