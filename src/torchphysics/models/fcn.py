import torch
import torch.nn as nn

from .diffeqmodel import DiffEqModel


class BlockFCN(DiffEqModel):
    """A fully connected neural network with constant width.

    Parameters
    ----------
    variable_dims : dic
        A dictionary containing the dimensionality of the input variables.
        Gets automatically created by the setting and can be called over
        Setting.variable_dims. 
    solution_dims : dic
        A dictionary containing the dimensionality of the output functions.
        Gets automatically created by the setting and can be called over
        Setting.solution_dims. 
    blocks : int
        number of relu/tanh blocks in the FCN
        (no of hidden layers will be 2*blocks)
    width : int
        width of the hidden layers
    """

    def __init__(self, variable_dims, solution_dims, blocks=3, width=100):
        super().__init__(variable_dims=variable_dims,
                         solution_dims=solution_dims)
        self.blocks = blocks
        self.width = width

        # build model
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(self.input_dim, self.width))
        torch.nn.init.xavier_normal_(self.layers[-1].weight, gain=1.4142)

        self.layers.append(nn.LeakyReLU())
        for _ in range(blocks):
            self.layers.append(nn.Linear(self.width, self.width))
            torch.nn.init.xavier_normal_(self.layers[-1].weight, gain=1.4142)
            self.layers.append(nn.LeakyReLU())

            self.layers.append(nn.Linear(self.width, self.width))
            torch.nn.init.xavier_normal_(self.layers[-1].weight, gain=5/3)
            self.layers.append(nn.Tanh())

        self.layers.append(nn.Linear(self.width, self.output_dim))
        torch.nn.init.xavier_normal_(self.layers[-1].weight, gain=1)

    def serialize(self):
        dct = {}
        dct['name'] = 'BlockFCN'
        dct['input_dim'] = self.input_dim
        dct['blocks'] = self.blocks
        dct['width'] = self.width
        dct['output_dim'] = self.output_dim
        return dct

    def forward(self, input_dict):
        """Stacks all input variables into a single tensor and applies the
        PyTorch model.

        Parameters
        ----------
        input_dict : ordered dict
            The dictionary of variables that is handed to the model
            (e.g. by a dataloader).

        Returns
        -------
        x : ordered dict
            A dictionary containing the searched functions. The model itself gives a
            tensor (or batch of tensors) as an output. Depending on the initially 
            given solution_dims, the tensor will be split up on the dimension
            of the solution. 
        """
        # prepare input
        x = self._prepare_inputs(input_dict)
        # apply model
        for layer in self.layers:
            x = layer(x)
        return self._prepare_outputs(x)


class SimpleFCN(DiffEqModel):
    """A fully connected neural network with constant width. The layers will
    be initalized with a xavier-normal distribution.

    Parameters
    ----------
    variable_dims : dic
        A dictionary containing the dimensionality of the input variables.
        Gets automatically created by the setting and can be called over
        Setting.variable_dims. 
    solution_dims : dic
        A dictionary containing the dimensionality of the output functions.
        Gets automatically created by the setting and can be called over
        Setting.solution_dims. 
    depth : int
        number of hidden layers in the FCN
    width : int
        width of the hidden layers
    """

    def __init__(self, variable_dims, solution_dims, normalization_dict=None, depth=3,
                 width=20):
        super().__init__(variable_dims=variable_dims,
                         solution_dims=solution_dims,
                         normalization_dict=normalization_dict)

        self.depth = depth
        self.width = width

        # build model
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(self.input_dim, self.width))
        torch.nn.init.xavier_normal_(self.layers[-1].weight, gain=5.0/3.0)
        self.layers.append(nn.Tanh())

        for _ in range(depth):
            self.layers.append(nn.Linear(self.width, self.width))
            torch.nn.init.xavier_normal_(self.layers[-1].weight, gain=5.0/3.0)
            self.layers.append(nn.Tanh())

        self.layers.append(nn.Linear(self.width, self.output_dim))
        torch.nn.init.xavier_normal_(self.layers[-1].weight, gain=1)

    def serialize(self):
        dct = {}
        dct['name'] = 'SimpleFCN'
        dct['input_dim'] = self.input_dim
        dct['depth'] = self.depth
        dct['width'] = self.width
        dct['output_dim'] = self.output_dim
        return dct

    def forward(self, input_dict):
        """Stacks all input variables into a single tensor and applies
        the PyTorch model

        Parameters
        ----------
        input_dict : ordered dict
            The dictionary of variables that is handed to the model
            (e.g. by a dataloader).

        Returns
        -------
        x : ordered dict
            A dictionary containing the searched functions. The model itself gives a
            tensor (or batch of tensors) as an output. Depending on the initially 
            given solution_dims, the tensor will be split up on the dimension
            of the solution. 
        """
        # prepare input
        x = self._prepare_inputs(input_dict)
        # apply model
        for layer in self.layers:
            x = layer(x)
        return self._prepare_outputs(x)
