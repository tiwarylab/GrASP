from typing import Optional

import torch
from torch import Tensor
from torch_scatter import scatter_mean

from pytorch_geometric.nn.inits import ones, zeros
from torch.nn.functional import sigmoid



class GraphNormSigmoid(torch.nn.Module):
    """A modification of graph normalization as described in the
    `"GraphNorm: A Principled Approach to Accelerating Graph Neural Network
    Training" <https://arxiv.org/abs/2009.03294>`_ paper
    This version bounds the alpha parameter from 0 to 1.

    Args:
        in_channels (int): Size of each input sample.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
    """
    def __init__(self, in_channels: int, eps: float = 1e-5):
        super().__init__()

        self.in_channels = in_channels
        self.eps = eps

        self.weight = torch.nn.Parameter(torch.Tensor(in_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(in_channels))
        self.mean_scale = torch.nn.Parameter(torch.Tensor(in_channels))

        self.reset_parameters()


    def reset_parameters(self):
        ones(self.weight)
        zeros(self.bias)
        ones(self.mean_scale)


    def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        """"""
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        batch_size = int(batch.max()) + 1

        mean = scatter_mean(x, batch, dim=0, dim_size=batch_size)
        out = x - mean.index_select(0, batch) * sigmoid(self.mean_scale)
        var = scatter_mean(out.pow(2), batch, dim=0, dim_size=batch_size)
        std = (var + self.eps).sqrt().index_select(0, batch)
        return self.weight * out / std + self.bias


    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels})'


class AffineOnlyNorm(torch.nn.Module):
    """Rescale features with only learnable affine parameters.

    Args:
        in_channels (int): Size of each input sample.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
    """
    def __init__(self, in_channels: int, eps: float = 1e-5):
        super().__init__()

        self.in_channels = in_channels
        self.eps = eps

        self.weight = torch.nn.Parameter(torch.Tensor(in_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(in_channels))

        self.reset_parameters()


    def reset_parameters(self):
        ones(self.weight)
        zeros(self.bias)


    def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        """"""
        return self.weight * x + self.bias


    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels})'
