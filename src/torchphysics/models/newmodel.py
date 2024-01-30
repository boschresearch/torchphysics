import torch
import torch.nn as nn

from ..utils.user_fun import UserFunction
from ..problem.spaces import Points, Space


class ModelBlock(nn.Module):
    """
    Building block to construct arbritrary model architectures.

    Parameters
    ----------
    input_neurons: int
        The number of input neurons.
    output_neurons: int
        The number of output neurons.
    needs_point_input : bool, optional
        If this blocks needs information of the space to which the inputs belong.
        E.g. if a input transform like x -> x^2 and t -> t + 2 should be applied to the 
        data. 
    Note
    ----
    In the final neural network mulitple of these block will be stacked, so for two blocks that 
    are behind one another the in_neurons and out_neurons have to match.
    """
    def __init__(self, input_neurons, output_neurons, needs_point_input=False):
        super().__init__()
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.needs_point_input = needs_point_input


class Model(nn.Module):
    """Neural networks that can be trained to fulfill user-defined conditions.

    Parameters
    ----------
    input_space : Space
        The space of the points the can be put into this model.
    output_space : Space
        The space of the points returned by this model.
    """
    def __init__(self, input_space, output_space, *model_blocks : ModelBlock):
        super().__init__()
        self.input_space = input_space
        self.output_space = output_space
        for i in range(len(model_blocks)):
            if i == 0:
                assert self.input_space.dim == model_blocks[i].input_neurons, \
                "Space dimension and size of input layer do not match!"
            else:
                assert model_blocks[i-1].output_neurons == model_blocks[i].input_neurons, \
                "Input and output of the blocks, at position " + str(i) + ", do not match!"
            if i == len(model_blocks)-1:
                assert self.output_space.dim == model_blocks[i].output_neurons, \
                "Space dimension and size of output layer do not match!"

        self.building_blocks = nn.Sequential(*model_blocks)
    
    def _fix_points_order(self, points):
        if points.space != self.input_space:
            if points.space.keys() != self.input_space.keys():
                raise ValueError(f"""Points are in {points.space} but should lie
                                     in {self.input_space}.""")
            points = points[..., list(self.input_space.keys())]
        return points

    def forward(self, points):
        if not self.building_blocks[0].needs_point_input:
            points = self._fix_points_order(points)
        return Points(self.building_blocks(points), self.output_space)
    

class NormalizationLayer(ModelBlock):
    """
    A first layer that scales a domain to the range (-1, 1)^domain.dim, since this
    can improve convergence during training.

    Parameters
    ----------
    domain : Domain
        The domain from which this layer expects sampled points. The layer will use
        its bounding box to compute the normalization factors.
    """
    def __init__(self, domain):
        super().__init__(input_neurons=domain.space.dim, output_neurons=domain.space.dim)
        self.normalize = nn.Linear(domain.space.dim, domain.space.dim)

        box = domain.bounding_box()
        mins = box[::2]
        maxs = box[1::2]

        # compute width and center
        diag = []
        bias = []
        for i in range(domain.dim):
            diag.append(maxs[i] - mins[i])
            bias.append((maxs[i] + mins[i])/2)
        
        diag = 2./torch.tensor(diag)
        bias = -torch.tensor(bias)*diag
        with torch.no_grad():
            self.normalize.weight.copy_(torch.diag(diag))
            self.normalize.bias.copy_(bias)

    def forward(self, points):
        return self.normalize(points)


def _construct_FC_layers(hidden, activations, xavier_gains, include_activation_in_last):
    """Constructs the layer structure for a fully connected neural network.
    """
    if not isinstance(activations, (list, tuple)):
        activations = len(hidden) * [activations]
    if not isinstance(xavier_gains, (list, tuple)):
        xavier_gains = len(hidden) * [xavier_gains]

    layers = []
    for i in range(len(hidden)-1):
        layers.append(nn.Linear(hidden[i], hidden[i+1]))
        torch.nn.init.xavier_normal_(layers[-1].weight, gain=xavier_gains[i])
        if i < len(hidden) - 1:
            layers.append(activations[i])
        # maybe in last layer no activation is wanted:
        elif include_activation_in_last:
            layers.append(activations[i])
    return layers


class FCN(ModelBlock):
    """A simple fully connected neural network.

    Parameters
    ----------
    layer_structure: list or tuple
        The number and size of the layers of the neural network.
        The lenght of the list/tuple will be equal to the number
        of layers, while the i-th entry will determine the number
        of neurons of each layer.
        E.g hidden = (10, 5) -> 2 layers, with 10 and 5 neurons.
    activations : torch.nn or list, optional
        The activation functions of this network. If a single function is passed
        as an input, will use this function for each layer.
        If a list is used, will use the i-th entry for i-th layer.
        Deafult is nn.Tanh().
    xavier_gains : float or list, optional
        For the weight initialization a Xavier/Glorot algorithm will be used.
        The gain can be specified over this value.
        Default is 5/3. 
    activation_in_last_layer : bool, optional
        If the last layer of this FCN should apply the activation function or not.
        Default is False.
    """
    def __init__(self, layer_structure=(20,20,20),
                 activations=nn.Tanh(), xavier_gains=5/3, 
                 activation_in_last_layer=False):
        super().__init__(input_neurons=layer_structure[0], output_neurons=layer_structure[-1])

        layers = _construct_FC_layers(hidden=layer_structure, 
                                      activations=activations, xavier_gains=xavier_gains, 
                                      include_activation_in_last=activation_in_last_layer)

        self.sequential = nn.Sequential(*layers)

    def forward(self, points):
        return self.sequential(points)


class InputTransform(ModelBlock):
    """
    A first layer that applies a customizable transformation on the input.
    For example the domain contains the variable x, but as an input both x and x^2 should
    be used. To accomplish this one can define a function f, which gets x as an input 
    and outputs a concatinaton of x and x^2:
    
        def f(x): return torch.column_stack((x, x**2))
    
    Only the output of f will be used as an input for the following network pieces.
    Other transforms can be utilized in the same way.
    This can improve convergence and accruacy, since the neural network has a richer input
    information. 

    Parameters
    ----------
    transform_fn : callable
        The function that applies the transformation of the domain variables.
    output_size : int
        The dimension of the input of the transform function. 
        (In the example above = 1 if x is one dimensional or = 2 if x is two dimensional)
    output_size : int
        The number of output that the transform function returns. 
        (In the example above = 2 if x is one dimensional or = 4 if x is two dimensional)
    """ 
    def __init__(self, transform_fn, input_size, output_size):
        if not isinstance(transform_fn, UserFunction):
            transform_fn = UserFunction(transform_fn)
        self.transform_fn = transform_fn 

        super().__init__(input_neurons=input_size, 
                         output_neurons=output_size, 
                         needs_point_input=True)

    def forward(self, points):
        return self.transform_fn(points)


class Parallel(Model):
    """A model that wraps multiple models which should be applied in parallel.

    Parameters
    ----------
    *models :
        The models that should be evaluated parallel. The evaluation
        happens in the order that the models are passed in.
        The outputs of the models will be concatenated. 
        The models are not allowed to have the same output spaces, but can
        have the same input spaces.
    """
    def __init__(self, *models):
        input_space = Space({})
        output_space = Space({})
        for model in models:
            assert output_space.keys().isdisjoint(model.output_space)
            input_space = input_space * Space(model.input_space - input_space)
            output_space = output_space * model.output_space
        super().__init__(input_space, output_space)
        self.models = nn.ModuleList(models)
    
    def forward(self, points):
        out = []
        for model in self.models:
            out.append(model(points[..., list(model.input_space.keys())]))
        return Points.joined(*out)


class AdaptiveWeightLayer(nn.Module):
    """
    Adds adaptive weights to the non-reduced loss. The weights are maximized by
    reversing the gradients, similar to the idea in [1].
    Should currently only be used with fixed points.

    Parameters
    ----------
    n : int
        The amount of sampled points in each batch.

    Notes
    -----
    ..  [1] L. McClenny, "Self-Adaptive Physics-Informed Neural Networks using a Soft
        Attention Mechanism", 2020.
    """
    class GradReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.neg()

    @classmethod
    def grad_reverse(cls, x):
        return cls.GradReverse.apply(x)

    def __init__(self, n):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(n)
        )

    def forward(self, points):
        weight = self.grad_reverse(self.weight)
        return weight*points
