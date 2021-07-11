from typing import Sequence
import torch
import numpy as np


def Parameter(init, name, shape=(1,)):
    if isinstance(init, torch.Tensor):
        data = init
    elif isinstance(init, np.ndarray):
        data = torch.from_numpy(init)
    elif isinstance(init, Sequence):
        data = torch.Tensor(init)
    elif isinstance(init, float):
        data = torch.Tensor((init,))
    elif isinstance(init, int):
        data = torch.Tensor((float(init),))
    elif init == 'normal':
        data = torch.randn(*shape)
    else:
        raise ValueError(f"'init' should not be {type(init)}.")

    param = ParameterSub(data)
    param.set_name(name)
    return param


class ParameterSub(torch.nn.Parameter):
    """
    A parameter that is part of the model and can be learned during training.

    Parameters
    ----------
    name : str
        Name of this parameter. By this name, the parameter will be found in the
        setting.
    """
    def set_name(self, name):
        self._name = name

    def get_name(self):
        return self._name
