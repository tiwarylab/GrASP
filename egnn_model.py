from torch import nn
import torch
import math

import torch.nn as nn
from torch_geometric.nn.norm import InstanceNorm
from model import GAT_block
from torch_geometric.nn import GATConv, GATv2Conv

"""Adapted from https://github.com/arneschneuing/DiffSBDD. 
Equivariant updates follow the orignal paper, latent updates use GASP."""

class EquivariantUpdate(nn.Module):
    def __init__(self, hidden_dim, normalization_factor, aggregation_method,
                 edges_in_d=1, act_fn=nn.SiLU(), tanh=False, coords_range=10.0):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        input_edge = hidden_dim * 2 + edges_in_d
        layer = nn.Linear(hidden_dim, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            layer)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr, edge_mask, update_coords_mask=None):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        if self.tanh:
            trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
        if edge_mask is not None:
            trans = trans * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)

        if update_coords_mask is not None:
            agg = update_coords_mask * agg

        coord = coord + agg
        return coord

    def forward(self, h, coord, edge_index, coord_diff, edge_attr=None,
                node_mask=None, edge_mask=None, update_coords_mask=None):
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask,
                                 update_coords_mask=update_coords_mask)
        if node_mask is not None:
            coord = coord * node_mask
        return coord


class EquivariantBlock(nn.Module):
    def __init__(self, hidden_dim, edge_feat_nf=7, act_fn=nn.SiLU(), n_layers=1,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, egnn_aggr='sum', GAT_aggr='multi', GAT_heads=4,
                 GAT_drop_prob=0.1, GAT_fill_value=torch.Tensor([0,0,0,0,0,0,1]), GAT_style=GATv2Conv):
        super(EquivariantBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.egnn_aggr = egnn_aggr
        self.GAT_aggr = GAT_aggr
        self.GAT_fill_value = GAT_fill_value
        self.GAT_style = GAT_style
        self.GAT_drop_prob = GAT_drop_prob

        # GASP Blocks
        for i in range(0, n_layers):
            self.add_module("gasp_%d" % i, GAT_block(hidden_dim, hidden_dim, GAT_heads, edge_dim=edge_feat_nf, drop_prob=GAT_drop_prob,
                                                     GAT_aggr=self.GAT_aggr, GAT_fill_value=self.GAT_fill_value,
                                                     GAT_style=self.GAT_style))
        self.add_module("gcl_equiv", EquivariantUpdate(hidden_dim, edges_in_d=edge_feat_nf+1, act_fn=act_fn, tanh=tanh,
                                                       coords_range=self.coords_range_layer,
                                                       normalization_factor=self.normalization_factor,
                                                       aggregation_method=self.egnn_aggr))
        self.to(self.device)

    def forward(self, h, x, edge_index, edge_attr=None, node_mask=None, edge_mask=None, update_coords_mask=None):
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        
        for i in range(0, self.n_layers):
            h = self._modules["gasp_%d" % i](h, edge_index, edge_attr=edge_attr)

        x = self._modules["gcl_equiv"](h, x, edge_index, coord_diff, edge_attr=torch.cat([edge_attr, distances], dim=-1),
                                       node_mask=node_mask, edge_mask=edge_mask, update_coords_mask=update_coords_mask)

        if node_mask is not None:
            h = h * node_mask
        return h, x


class GASP_EGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2, n_layers=12, GAT_heads=4,
                 GAT_drop_prob=0.1, GAT_aggr='multi', GAT_fill_value=torch.Tensor([0,0,0,0,0,0,1]), 
                 GAT_style=GATv2Conv, act_fn=nn.SiLU(), norm_diff=True, tanh=False, coords_range=15, 
                 norm_constant=1, inv_sublayers=1, sin_embedding=False, normalization_factor=100, egnn_aggr='sum'):
        super(GASP_EGNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range/n_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.egnn_aggr = egnn_aggr
        self.GAT_aggr = GAT_aggr
        self.GAT_heads = GAT_heads
        self.GAT_drop_prob = GAT_drop_prob
        self.GAT_fill_value = GAT_fill_value
        self.GAT_style = GAT_style

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = len(self.GAT_fill_value)

        # GASP Encoder
        self.encoder = nn.Sequential( 
            InstanceNorm(input_dim, affine=True),
            nn.Linear(input_dim, self.hidden_dim, bias=False),
            InstanceNorm(self.hidden_dim, affine=True),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
            InstanceNorm(self.hidden_dim, affine=True),
            nn.ELU()
        )

        for i in range(0, n_layers):
            self.add_module("e_block_%d" % i, EquivariantBlock(hidden_dim, edge_feat_nf=edge_feat_nf,
                                                               act_fn=act_fn, n_layers=inv_sublayers,
                                                               norm_diff=norm_diff, tanh=tanh,
                                                               coords_range=coords_range, norm_constant=norm_constant,
                                                               sin_embedding=self.sin_embedding,
                                                               normalization_factor=self.normalization_factor,
                                                               egnn_aggr=self.egnn_aggr, GAT_aggr=self.GAT_aggr,
                                                               GAT_heads=self.GAT_heads, GAT_drop_prob=self.GAT_drop_prob,
                                                               GAT_fill_value=self.GAT_fill_value, 
                                                               GAT_style=self.GAT_style))

        cat_dim = hidden_dim * (n_layers + 1)

        self.decoder = nn.Sequential(
            nn.Linear(cat_dim, 4*hidden_dim),
            nn.ELU(),
            nn.Linear(4*hidden_dim, 2*hidden_dim),
            nn.ELU(),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.ELU(),
            nn.Linear(int(hidden_dim/2), int(hidden_dim/4)),
            nn.ELU(),
            nn.Linear(int(hidden_dim/4), output_dim)
        )
        
        self.recon = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, input_dim)
        )

        self.to(self.device)

    def forward(self, input, node_mask=None, edge_mask=None, update_coords_mask=None):
        h = input.x
        x = input.coords
        edge_index = input.edge_index
        edge_attr = input.edge_attr
        
        h = self.encoder(h)
        jk_inputs = [h]

        for i in range(0, self.n_layers):
            h, x = self._modules["e_block_%d" % i](
                h, x, edge_index, edge_attr=edge_attr, node_mask=node_mask, 
                edge_mask=edge_mask, update_coords_mask=update_coords_mask)
            jk_inputs.append(h)

        message_out = h

        h = torch.cat(jk_inputs, dim=-1)
        h = self.decoder(h)

        reconstruction = self.recon(message_out)

        if node_mask is not None:
            h = h * node_mask
            
        return h, x, reconstruction


class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies)/max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff


def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result
