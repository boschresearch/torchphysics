import torch
import torch.nn as nn

from .diffeqmodel import DiffEqModel


class SimpleFCN(DiffEqModel):
    """A fully connected neural network with constant width.

    Parameters
    ----------
    input_dim : int
        dimensionality of the input variable
    depth : int
        number of hidden layers in the FCN
    width : int
        width of the hidden layers
    output_dim : int
        amount of output neurons
    """

    def __init__(self, input_dim, blocks=3, width=100, output_dim=1):
        super().__init__()

        self.input_dim = input_dim
        self.blocks = blocks
        self.width = width
        self.output_dim = output_dim

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
        dct['name'] = 'SimpleFCN'
        dct['input_dim'] = self.input_dim
        dct['blocks'] = self.blocks
        dct['width'] = self.width
        dct['output_dim'] = self.output_dim
        return dct

    def forward(self, input_dict, track_gradients=True):
        """Stacks all input variables into a single tensor.

        Parameters
        ----------
        input_dict : ordered dict
            The dictionary of variables that is handed to the model
            (e.g. by a dataloader).
        track_gradients : bool or list of str or list of DiffVariables
            Whether the gradients w.r.t. the inputs should be tracked.
            Tracking can be necessary for training of a PDE.
            If True, all gradients will be tracked.
            If a list of strings or variables is passed, gradient is tracked
            only for the variables in the list.
            If False, no gradients will be tracked.

        Returns
        -------
        x : torch.Tensor
            Output of the model
        """
        # prepare input
        x = self._prepare_inputs(input_dict, track_gradients)
        # apply model
        for layer in self.layers:
            x = layer(x)
        return x
