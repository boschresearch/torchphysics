import torch
import torch.nn as nn
from .model import Model
from ..problem.spaces import Points

class DeepRitzNet(Model):
    """
    Implementation of the architecture used in the Deep Ritz paper [1].
    Consists of fully connected layers and residual connections.

    Parameters
    ----------
    input_space : Space
        The space of the points the can be put into this model.
    output_space : Space
        The space of the points returned by this model.
    width : int
        The width of the used hidden fully connected layers.
    depth : int
        The amount of subsequent residual blocks.

    Notes
    -----
    ..  [1] Weinan E and Bing Yu, "The Deep Ritz method: A deep learning-based numerical
        algorithm for solving variational problems", 2017
    """
    def __init__(self, input_space, output_space, width, depth):
        super().__init__(input_space, output_space)
        self.width = width
        self.depth = depth
        self.linearIn = nn.Linear(self.input_space.dim, self.width)
        self.linear1 = nn.ModuleList()
        self.linear2 = nn.ModuleList()
        for _ in range(self.depth):
            self.linear1.append(nn.Linear(self.width, self.width))
            self.linear2.append(nn.Linear(self.width, self.width))

        self.linearOut = nn.Linear(self.width, self.output_space.dim)

    def forward(self, x):
        x = self._fix_points_order(x)
        x = self.linearIn(x) # Match input dimension of network
        for (layer1,layer2) in zip(self.linear1, self.linear2):
            x_temp = torch.relu(layer1(x)**3)
            x_temp = torch.relu(layer2(x_temp)**3)
            x = x_temp + x
        
        return Points(self.linearOut(x), self.output_space)