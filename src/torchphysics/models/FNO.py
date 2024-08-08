import torch
import torch.nn as nn
from .model import Model
from ..problem.spaces import Points


class _FourierLayer(nn.Model):
    """Implements a single fourier layer of the FNO from [1]. Is of the form:

    Parameters
    ----------
    mode_num : int, tuple
        The number of modes that should be used. For resolutions with higher
        frequenzies, the layer will discard everything above `mode_num` and
        in the inverse Fourier transform append zeros. In higher dimensional
        data, a tuple can be passed in with len(mode_num) = dimension.
    in_features : int
        size of each input sample.

    Notes
    -----
    ..  [1]
    """

    def __init__(self, mode_num, in_features, xavier_gain):
        # Transform mode_num to tuple:
        if isinstance(mode_num, int):
            mode_num = (mode_num,)

        super().__init__()
        self.mode_num = torch.tensor(mode_num)
        self.in_features = in_features
        # self.linear_weights = torch.nn.Linear(in_features=in_features,
        #                                        out_features=in_features,
        #                                        bias=False)

        self.fourier_weights = torch.nn.Parameter(
            torch.empty((in_features, *self.mode_num)), dtype=torch.complex32
        )
        torch.nn.init.xavier_normal_(self.fourier_weights, gain=xavier_gain)

    def forward(self, points):
        ### Linear skip connection
        # linear_out = self.linear_weights(points)
        ### Fourier part
        # Computing how much each dimension has to cut/padded:
        # Here we need that points.shape = (batch, data_dim, resolution)
        padding = torch.zeros(
            2 * len(self.mode_num), device=points.device, dtype=torch.int32
        )
        padding[1::2] = torch.flip(
            (self.mode_num - torch.tensor(points.shape[2:])), dims=(0,)
        )
        fft = torch.nn.functional.pad(
            torch.fft.fftn(points, dim=len(self.mode_num), norm="ortho"),
            padding.tolist(),
        )  # here remove to high freq.
        weighted_fft = self.fourier_weights * fft
        ifft = torch.fft.ifftn(
            torch.nn.functional.pad(
                weighted_fft, (-padding).tolist()
            ),  # here add high freq.
            dim=len(self.mode_num),
            norm="ortho",
        )
        ### Connect linear and fourier output
        return ifft

    @property
    def in_features(self):
        return self.in_features

    @property
    def out_features(self):
        return self.in_features


class FNO(Model):

    def __init__(
        self,
        input_space,
        output_space,
        upscale_size,
        fourier_layers,
        fourier_modes,
        activations,
        xavier_gains,
    ):
        super().__init__(input_space, output_space)
