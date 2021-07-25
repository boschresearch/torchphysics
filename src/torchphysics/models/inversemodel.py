'''Contains a Wrapper for all newtworks to use them in inverse problems

import torch
import numpy as np
from . import DiffEqModel

class InverseModel(DiffEqModel):
    """A class to wrap other DiffEqModels for the inverse problem.

    Parameters
    ----------
    model : nn.Module
        The network that should approximate the function.
    params : dic
        A dictonary containing the name and either a initial guess or the dimensions 
        of the variables. The inital guess can be a torch.tensor, list or numpy array.
        If a dimensions is given it has to be a int or tuple.
    """
    def __init__(self, model, params):
        super().__init__()
        self.model = model
        self.add_parameters(params)

    def add_parameters(self, params):
        for pname, v in params.items():
            if isinstance(v, torch.Tensor):
                param = torch.nn.Parameter(v)
            elif isinstance(v, (int, tuple)):
                param = torch.nn.Parameter(torch.zeros(v))
            elif isinstance(v, (list, np.array)):
                param = torch.nn.Parameter(torch.FloatTensor(v))
            else:
                raise ValueError('Expected the parameters to be tensors, int, tuple,'
                                 + ' list or numpy.array. Found: ' + type(v))
            self.model.register_parameter(name=pname, param=param)

    def get_parameters(self):
        return self.model._parameters

    def forward(self, input_dict):
        return self.model.forward(input_dict)

    def serialize(self):
        dct = self.model.serialize()
        #dct['parameters'] = self.model._parameters
        return dct
'''