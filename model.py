import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.norm import InstanceNorm
from torch_geometric.nn.aggr import MultiAggregation

class GAT_model(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2, weight_groups=1, group_layers=1,
     GAT_heads=4, drop_prob=0.1, GAT_aggr="mean", GAT_fill_value=torch.Tensor([0,0,0,0,0,0,1]),
      GAT_style=GATv2Conv): # hidden_dim must be divisible by 8
        super(GAT_model, self).__init__()
        self.weight_groups = weight_groups
        self.edge_dim = len(GAT_fill_value)

        self.pre_norm = InstanceNorm(input_dim, affine=True)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            InstanceNorm(hidden_dim, affine=True),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            InstanceNorm(hidden_dim, affine=True),
            nn.ELU()
        )

        self.message_layers = torch.nn.ModuleList()
        for i in range(group_layers):
            self.message_layers.append(GAT_block(hidden_dim, hidden_dim, 
            GAT_heads=GAT_heads, edge_dim=self.edge_dim, drop_prob=drop_prob, GAT_aggr=GAT_aggr,
            GAT_fill_value=GAT_fill_value, GAT_style=GAT_style))

        cat_dim = hidden_dim * (weight_groups*group_layers + 1)

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

    def forward(self, input):
        x = input.x
        x = self.pre_norm(x)
        x = self.encoder(x)
        jk_inputs = [x]

        for i in range(self.weight_groups): # weight sharing / refinement
            for message_layer in self.message_layers:
                x = message_layer(x, input.edge_index, input.edge_attr)
                jk_inputs.append(x)

        message_out = x

        x = torch.cat(jk_inputs, dim=-1) # jumping knowledge skipcon
        x = self.decoder(x)

        reconstruction = self.recon(message_out)

        return x, reconstruction


class GAT_block(nn.Module):
    def __init__(self, input_dim, output_dim, GAT_heads, edge_dim, drop_prob,
     GAT_aggr, GAT_fill_value, GAT_style):
        super(GAT_block, self).__init__()
        head_dim = int(output_dim/GAT_heads)

        if GAT_aggr == "multi":
            GAT_aggr = MultiAggregation(['mean', 'sum'], mode='proj',
             mode_kwargs={"in_channels":head_dim, "out_channels":head_dim})

        self.GAT = GAT_style(input_dim, head_dim,
         heads=GAT_heads, edge_dim=edge_dim, bias=False, dropout=drop_prob,
          aggr=GAT_aggr, fill_value=GAT_fill_value)

        self.norm = InstanceNorm(output_dim, affine=True)
        self.elu = nn.ELU()

    def forward(self, x, edge_index, edge_attr):
        GAT_out = self.GAT(x, edge_index, edge_attr)
        norm_out = self.norm(GAT_out)
        add_out = self.elu(torch.add(x, norm_out))

        return add_out

