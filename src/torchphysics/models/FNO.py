import torch
import torch.nn as nn
from .model import Model

from ..problem.spaces import Points


class _FourierLayer(nn.Module):
    """Implements a single Fourier layer of the FNO. For the parameter description see
    the FNO documentation.
    """
    def __init__(self, channels, mode_num, 
                 linear_connection : bool = False, skip_connection : bool = False,
                 bias : bool = False, xavier_gain=5.0/3.0, space_res=None):
        # Transform mode_num to tuple:
        if isinstance(mode_num, int):
            mode_num = (mode_num,)

        super().__init__()
        
        self.channels = channels
        self.skip_connection : bool = skip_connection
        # Values for Fourier transformation
        self.mode_num = torch.tensor(mode_num)
        self.data_dim = len(mode_num)
        self.fourier_dims = list(range(1, self.data_dim+1))
        
        # Learnable parameters
        self.fourier_kernel = torch.nn.Parameter(
           torch.empty((*self.mode_num, self.channels), dtype=torch.cfloat))
        torch.nn.init.xavier_normal_(self.fourier_kernel, gain=xavier_gain)

        self.linear_connection : bool = linear_connection
        if self.linear_connection:
            self.linear_transform = nn.Linear(channels, channels, bias=bias)        

        self.use_bn : bool = False
        if space_res:
            self.use_bn = True
            if self.data_dim == 1:
                self.bn = torch.nn.BatchNorm1d(space_res)
            else:
                raise NotImplementedError(f"Dimension {self.data_dim} currently not \
                                          supported for batch normalization")
        

    def forward(self, points):
        fft = torch.fft.rfftn(points, dim=self.fourier_dims)

        # Next add zeros or remove fourier modes to fit input for wanted freq.
        original_fft_shape = torch.tensor(fft.shape[1:-1])
        # padding needs to extra values, since the torch.nn.functional.pad starts
        # from the last dimension (the channels in our case), there we dont need to 
        # change anything so only zeros in the padding.
        padding = torch.zeros(2*self.data_dim + 2, device=points.device, dtype=torch.int32)
        padding[3::2] = torch.flip((self.mode_num - original_fft_shape), dims=(0,))

        fft = torch.nn.functional.pad(fft, padding.tolist())
    
        fft *= self.fourier_kernel

        ifft = torch.fft.irfftn(fft, s=points.shape[1:-1], dim=self.fourier_dims)

        if self.linear_connection:
            ifft += self.linear_transform(points)

        if self.skip_connection:
            ifft += points

        if self.use_bn:
            return self.bn(ifft)

        return ifft
    

class FNO(Model):
    """ The Fourier Neural Operator original developed in [1].

    Parameters
    ----------
    input_space : Space
        The space of the points the can be put into this model.
    output_space : Space
        The space of the points returned by this model.
    fourier_layers : int
        The number of fourier layers of this network. Each fourier layer consists
        of a spectral convolution with learnable kernels. See [1] for an overview 
        of the model. Linear transformations and skip connections can be enabled 
        in each layer as well.
    hidden_channles : int
        The number of hidden channels.
    fourier_modes : int or list, tuple
        The number of Fourier modes that will be used for the spectral convolution
        in each layer. Modes over the given value will be truncated, and in case
        of not enough modes they are padded with 0.
        In case of a 1D space domain you can pass in one integer or a list of
        integers, such that in each layer a different amount of modes is used.
        In case of a N-dimensional space domain a list (or tuple) of N numbers
        must be passed in (Setting the modes for each direction), or again
        a list of list containig each N numbers to vary the modes per layer.
    activations : torch.nn or list, tuple
        The activation function after each Fourier layer.
        Default is torch.nn.Tanh()
    skip_connections : bool or list, tuple
        If a skip connection is enabled in each Fourier layer, adding the original
        input of the layer to the output without any transformations.
    linear_connection : bool or list, tuple
        If the input of each Fourier layer should also be transformed by a
        (learnable) linear mapping and added to the output.
    bias : bool or list, tuple
        If the above linear connection should include a (learnable) bias vector.
    channel_up_sample_network : torch.nn
        The network that transforms the input channel dimension to the 
        hidden channel dimension. (The mapping P in [1], Figure 2)
        Default is a linear mapping.
    channel_down_sample_network : torch.nn
        The network that transforms the hidden channel dimension to the 
        output channel dimension. (The mapping Q in [1], Figure 2)
        Default is a linear mapping.
    xavier_gains : int or list, tuple
        For the weight initialization a Xavier/Glorot algorithm will be used.
        The gain can be specified over this value.
        Default is 5/3.
    space_resolution : int or None
        The resolution of the space grid used for training. This value is optional.
        If specified, a batch normalization over the space dimension will be applied
        in each Fourier layer. This leads to smoother solutions and better local
        approximations. But (currently) removes the super resolution property of the 
        FNO. This is currently only possible for 1D space dimensions.

    Notes
    -----
    The FNO assumes that the data is of the shape 
        (batch, space_dim_1, ..., space_dim_n, channels).
    E.g. for a one dimensional problem we have (batch, grid points, channels).
    Additionally, the data needs to exists on a uniform grid to accurately 
    compute the Fourier transformation.
    
    Note, this networks assumes that the input and output are real numbers.
    It does not work in the case of complex numbers.

    ..  [1] Zong-Yi Li et al., "Fourier Neural Operator for Parametric Partial 
            Differential Equations", 2020
    """
    def __init__(self, input_space, output_space, fourier_layers : int, 
                 hidden_channels : int = 16, fourier_modes = 16, activations=torch.nn.Tanh(), 
                 skip_connections = False, linear_connections = True, bias = True,
                 channel_up_sample_network = None, channel_down_sample_network = None,
                 xavier_gains=5.0/3, space_resolution = None):
        super().__init__(input_space, output_space)

        # Transform data to list values for each layer:
        skip_connections = self._extend_data(fourier_layers, skip_connections)
        bias = self._extend_data(fourier_layers, bias)
        linear_connections = self._extend_data(fourier_layers, linear_connections)
        activations = self._extend_data(fourier_layers, activations)
        xavier_gains = self._extend_data(fourier_layers, xavier_gains)

        if isinstance(fourier_modes, int):
            fourier_modes = fourier_layers * [fourier_modes]
        elif isinstance(fourier_modes, (list, tuple)):
            if len(fourier_modes) < fourier_layers:
                fourier_modes = fourier_layers * [fourier_modes] 
        else:
            raise ValueError(f"Invalid input for fourier modes")

        # Define network architecture
        layers = []

        in_channels = self.input_space.dim
        out_channels = self.output_space.dim

        if not channel_up_sample_network:
            self.channel_up_sampling = nn.Linear(in_channels, 
                                                 hidden_channels, 
                                                 bias=True)
        else:
            self.channel_up_sampling = channel_up_sample_network

        if not channel_down_sample_network:
            self.channel_down_sampling = nn.Linear(hidden_channels, 
                                                   out_channels, 
                                                   bias=True)
        else:
            self.channel_down_sampling = channel_down_sample_network

        for i in range(fourier_layers):
            new_layer = _FourierLayer(hidden_channels, fourier_modes[i], 
                                      linear_connections[i],
                                      skip_connections[i], bias[i], 
                                      xavier_gains[i],
                                      space_res=space_resolution)
            layers.append(new_layer)
            layers.append(activations[i])

        self.fourier_sequential = nn.Sequential(*layers)


    def _extend_data(self, fourier_layers, skip_connections):
        if not isinstance(skip_connections, (list, tuple)):
            skip_connections = fourier_layers * [skip_connections]
        return skip_connections


    def forward(self, points):
        points = self._fix_points_order(points)
        points_up_sampled = self.channel_up_sampling(points)
        fourier_points = self.fourier_sequential(points_up_sampled)
        output = self.channel_down_sampling(fourier_points)
        return Points(output, self.output_space)
