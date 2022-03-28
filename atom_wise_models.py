import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, GINConv
from torch_geometric.nn.norm import BatchNorm
import torch.nn.functional as F

#MLP Helper for GIN
class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction
        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)
        
class Two_Track_GIN_GAT_fixed_bn(nn.Module):
    def __init__(self, input_dim, output_dim=2, drop_prob=0.1, GAT_aggr="mean", GIN_aggr="add"):
        # No need for bias in GAT Convs due to batch norms
        super(Two_Track_GIN_GAT_fixed_bn, self).__init__() 
        self.preprocess1 = nn.Linear(input_dim, 72, bias=False)
        self.pre_BN1 = BatchNorm(72, track_running_stats=False)
        self.preprocess2 = nn.Linear(72, 64, bias=False)
        self.pre_BN2 = BatchNorm(64, track_running_stats=False)
        
        # Left Track
        self.left_GAT_1 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=GAT_aggr)
        self.left_BN1 = BatchNorm(64, track_running_stats=False)
        
        self.left_GAT_2 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=GAT_aggr)
        self.left_BN2 = BatchNorm(64, track_running_stats=False)

        self.left_GAT_3 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=GAT_aggr)
        self.left_BN3 = BatchNorm(64, track_running_stats=False)

        self.left_GAT_4 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=GAT_aggr)
        self.left_BN4 = BatchNorm(64, track_running_stats=False)
         
        # Right Track
        self.right_GAT_1 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
        self.right_BN1 = BatchNorm(64, track_running_stats=False)
        
        self.right_GAT_2 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
        self.right_BN2 = BatchNorm(64, track_running_stats=False)

        self.right_GAT_3 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
        self.right_BN3 = BatchNorm(64, track_running_stats=False)

        self.right_GAT_4 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
        self.right_BN4 = BatchNorm(64, track_running_stats=False)

        self.post_BN = BatchNorm(576, track_running_stats=False)
        self.postprocess1 = nn.Linear(576, 256)
        self.postprocess2 = nn.Linear(256, 128)
        self.postprocess3 = nn.Linear(128, 64)
        self.postprocess4 = nn.Linear(64, 32)
        self.postprocess5 = nn.Linear(32, 16)
        self.postprocess6 = nn.Linear(16, output_dim)

        self.elu = torch.nn.ELU()
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, input):
        x = self.pre_BN1(self.preprocess1(input.x))
        x = self.elu(x)
        x = self.pre_BN2(self.preprocess2(x))
        x = self.elu(x)
        
        # Left Track
        left_block_1_out = self.left_GAT_1(x, input.edge_index)
        left_block_1_out = self.elu(self.left_BN1(torch.add(left_block_1_out, x)))

        left_block_2_out = self.left_GAT_2(left_block_1_out,input.edge_index)
        left_block_2_out = self.elu(self.left_BN2(torch.add(left_block_2_out, left_block_1_out)))              # Resnet style skip connection
        
        left_block_3_out = self.left_GAT_3(left_block_2_out,input.edge_index)
        left_block_3_out = self.elu(self.left_BN3(torch.add(left_block_3_out, left_block_2_out)))

        left_block_4_out = self.left_GAT_4(left_block_3_out,input.edge_index)
        left_block_4_out = self.elu(self.left_BN4(torch.add(left_block_4_out, left_block_3_out)))
        
        # Right Track
        right_block_1_out = self.right_GAT_1(x, input.edge_index)
        right_block_1_out = self.elu(self.right_BN1(torch.add(right_block_1_out, x)))

        right_block_2_out = self.right_GAT_2(right_block_1_out,input.edge_index)
        right_block_2_out = self.elu(self.right_BN2(torch.add(right_block_2_out, right_block_1_out)))              # Resnet style skip connection
        
        right_block_3_out = self.right_GAT_3(right_block_2_out,input.edge_index)
        right_block_3_out = self.elu(self.right_BN3(torch.add(right_block_3_out, right_block_2_out)))

        right_block_4_out = self.right_GAT_4(right_block_3_out,input.edge_index)
        right_block_4_out = self.elu(self.right_BN4(torch.add(right_block_4_out, right_block_3_out)))
        
        combined = torch.cat((left_block_4_out,right_block_4_out,left_block_3_out,right_block_3_out,left_block_2_out,right_block_2_out,left_block_1_out,right_block_1_out,x), dim=-1)

        combined = self.post_BN(combined)
        
        x = self.elu(self.postprocess1(combined))
        x = self.elu(self.postprocess2(x))
        x = self.elu(self.postprocess3(x))
        x = self.elu(self.postprocess4(x))
        x = self.elu(self.postprocess5(x))
        x = self.postprocess6(x)

        return x

class Two_Track_GIN_GAT_Noisy_Nodes(nn.Module):
    def __init__(self, input_dim, output_dim=2, drop_prob=0.1, GAT_aggr="mean", GIN_aggr="add", noise_variance=0.02):
        self.noise_variance = noise_variance
        # No need for bias in GAT Convs due to batch norms
        super(Two_Track_GIN_GAT_Noisy_Nodes, self).__init__()
        self.BN0 = BatchNorm(input_dim, track_running_stats=False, affine=False)

        self.preprocess1 = nn.Linear(input_dim, 72, bias=False)
        self.pre_BN1 = BatchNorm(72, track_running_stats=False)
        self.preprocess2 = nn.Linear(72, 64, bias=False)
        self.pre_BN2 = BatchNorm(64, track_running_stats=False)

        # Left Track
        self.left_GAT_1 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=GAT_aggr)
        self.left_BN1 = BatchNorm(64, track_running_stats=False)

        self.left_GAT_2 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=GAT_aggr)
        self.left_BN2 = BatchNorm(64, track_running_stats=False)

        self.left_GAT_3 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=GAT_aggr)
        self.left_BN3 = BatchNorm(64, track_running_stats=False)

        self.left_GAT_4 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=GAT_aggr)
        self.left_BN4 = BatchNorm(64, track_running_stats=False)

        # Right Track
        self.right_GAT_1 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
        self.right_BN1 = BatchNorm(64, track_running_stats=False)

        self.right_GAT_2 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
        self.right_BN2 = BatchNorm(64, track_running_stats=False)

        self.right_GAT_3 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
        self.right_BN3 = BatchNorm(64, track_running_stats=False)

        self.right_GAT_4 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
        self.right_BN4 = BatchNorm(64, track_running_stats=False)

        self.post_BN = BatchNorm(576, track_running_stats=False)
        self.postprocess1 = nn.Linear(576, 256)
        self.postprocess2 = nn.Linear(256, 128)
        self.postprocess3 = nn.Linear(128, 64)
        self.postprocess4 = nn.Linear(64, 32)
        self.postprocess5 = nn.Linear(32, 16)
        self.postprocess6 = nn.Linear(16, output_dim)
        
        self.elu = torch.nn.ELU()
        self.softmax = torch.nn.Softmax(dim=0)

	# Regression head for reconstruction
        self.rcon1 = nn.Linear(128, 96)
        self.rcon2 = nn.Linear(96, 96)
        self.rcon3 = nn.Linear(96, input_dim)

    def forward(self, input):
        x = input.x
        if self.training:
            x += (x.std(dim=0)*self.noise_variance)*torch.randn_like(x)

        x = self.BN0(x)

        x = self.pre_BN1(self.preprocess1(x))
        x = self.elu(x)
        x = self.pre_BN2(self.preprocess2(x))
        x = self.elu(x)

        # Left Track
        left_block_1_out = self.left_GAT_1(x, input.edge_index)
        left_block_1_out = self.elu(self.left_BN1(torch.add(left_block_1_out, x)))

        left_block_2_out = self.left_GAT_2(left_block_1_out,input.edge_index)
        left_block_2_out = self.elu(self.left_BN2(torch.add(left_block_2_out, left_block_1_out)))              # Resnet style skip connection

        left_block_3_out = self.left_GAT_3(left_block_2_out,input.edge_index)
        left_block_3_out = self.elu(self.left_BN3(torch.add(left_block_3_out, left_block_2_out)))

        left_block_4_out = self.left_GAT_4(left_block_3_out,input.edge_index)
        left_block_4_out = self.elu(self.left_BN4(torch.add(left_block_4_out, left_block_3_out)))

        # Right Track
        right_block_1_out = self.right_GAT_1(x, input.edge_index)
        right_block_1_out = self.elu(self.right_BN1(torch.add(right_block_1_out, x)))

        right_block_2_out = self.right_GAT_2(right_block_1_out,input.edge_index)
        right_block_2_out = self.elu(self.right_BN2(torch.add(right_block_2_out, right_block_1_out)))              # Resnet style skip connection

        right_block_3_out = self.right_GAT_3(right_block_2_out,input.edge_index)
        right_block_3_out = self.elu(self.right_BN3(torch.add(right_block_3_out, right_block_2_out)))

        right_block_4_out = self.right_GAT_4(right_block_3_out,input.edge_index)
        right_block_4_out = self.elu(self.right_BN4(torch.add(right_block_4_out, right_block_3_out)))

        combined = torch.cat((left_block_4_out,right_block_4_out,left_block_3_out,right_block_3_out,left_block_2_out,right_block_2_out,left_block_1_out,right_block_1_out,x), dim=-1)

        combined = self.post_BN(combined)

        x = self.elu(self.postprocess1(combined))
        x = self.elu(self.postprocess2(x))
        x = self.elu(self.postprocess3(x))
        x = self.elu(self.postprocess4(x))
        x = self.elu(self.postprocess5(x))
        x = self.postprocess6(x)

	# Regression Head for Reconstruction Loss
        rcon_output = torch.cat((left_block_4_out, right_block_4_out), dim=-1)
        rcon_output = self.elu(self.rcon1(rcon_output))
        rcon_output = self.elu(self.rcon2(rcon_output))
        rcon_output = self.rcon3(rcon_output)

        return x, rcon_output


class Two_Track_GIN_GAT_Edge_Feat(nn.Module):
    def __init__(self, input_dim, output_dim=2, drop_prob=0.1, GAT_aggr="mean", GIN_aggr="add", noise_variance=0.02):
        self.noise_variance = noise_variance
        # No need for bias in GAT Convs due to batch norms
        super(Two_Track_GIN_GAT_Edge_Feat, self).__init__()
        self.BN0 = BatchNorm(input_dim, track_running_stats=False, affine=False)

        self.preprocess1 = nn.Linear(input_dim, 72, bias=False)
        self.pre_BN1 = BatchNorm(72, track_running_stats=False)
        self.preprocess2 = nn.Linear(72, 64, bias=False)
        self.pre_BN2 = BatchNorm(64, track_running_stats=False)

        # Left Track
        self.left_GAT_1 = GATv2Conv(64, 8, heads=8, edge_dim=6, bias=False, dropout=drop_prob, aggr=GAT_aggr)
        self.left_BN1 = BatchNorm(64, track_running_stats=False)

        self.left_GAT_2 = GATv2Conv(64, 8, heads=8, edge_dim=6, bias=False, dropout=drop_prob, aggr=GAT_aggr)
        self.left_BN2 = BatchNorm(64, track_running_stats=False)

        self.left_GAT_3 = GATv2Conv(64, 8, heads=8, edge_dim=6, bias=False, dropout=drop_prob, aggr=GAT_aggr)
        self.left_BN3 = BatchNorm(64, track_running_stats=False)

        self.left_GAT_4 = GATv2Conv(64, 8, heads=8, edge_dim=6, bias=False, dropout=drop_prob, aggr=GAT_aggr)
        self.left_BN4 = BatchNorm(64, track_running_stats=False)

        # Right Track
        self.right_GAT_1 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
        self.right_BN1 = BatchNorm(64, track_running_stats=False)

        self.right_GAT_2 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
        self.right_BN2 = BatchNorm(64, track_running_stats=False)

        self.right_GAT_3 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
        self.right_BN3 = BatchNorm(64, track_running_stats=False)

        self.right_GAT_4 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
        self.right_BN4 = BatchNorm(64, track_running_stats=False)

        self.post_BN = BatchNorm(576, track_running_stats=False)
        self.postprocess1 = nn.Linear(576, 256)
        self.postprocess2 = nn.Linear(256, 128)
        self.postprocess3 = nn.Linear(128, 64)
        self.postprocess4 = nn.Linear(64, 32)
        self.postprocess5 = nn.Linear(32, 16)
        self.postprocess6 = nn.Linear(16, output_dim)
        
        self.elu = torch.nn.ELU()
        self.softmax = torch.nn.Softmax(dim=0)

	# Regression head for reconstruction
        self.rcon1 = nn.Linear(128, 96)
        self.rcon2 = nn.Linear(96, 96)
        self.rcon3 = nn.Linear(96, input_dim)

    def forward(self, input):
        x = input.x
        if self.training:
            x += (x.std(dim=0)*self.noise_variance)*torch.randn_like(x)

        x = self.BN0(x)

        x = self.pre_BN1(self.preprocess1(x))
        x = self.elu(x)
        x = self.pre_BN2(self.preprocess2(x))
        x = self.elu(x)

        # Left Track
        left_block_1_out = self.left_GAT_1(x, input.edge_index, input.edge_attr)
        left_block_1_out = self.elu(self.left_BN1(torch.add(left_block_1_out, x)))

        left_block_2_out = self.left_GAT_2(left_block_1_out,input.edge_index, input.edge_attr)
        left_block_2_out = self.elu(self.left_BN2(torch.add(left_block_2_out, left_block_1_out)))              # Resnet style skip connection

        left_block_3_out = self.left_GAT_3(left_block_2_out,input.edge_index, input.edge_attr)
        left_block_3_out = self.elu(self.left_BN3(torch.add(left_block_3_out, left_block_2_out)))

        left_block_4_out = self.left_GAT_4(left_block_3_out,input.edge_index, input.edge_attr)
        left_block_4_out = self.elu(self.left_BN4(torch.add(left_block_4_out, left_block_3_out)))

        # Right Track
        right_block_1_out = self.right_GAT_1(x, input.edge_index)
        right_block_1_out = self.elu(self.right_BN1(torch.add(right_block_1_out, x)))

        right_block_2_out = self.right_GAT_2(right_block_1_out,input.edge_index)
        right_block_2_out = self.elu(self.right_BN2(torch.add(right_block_2_out, right_block_1_out)))              # Resnet style skip connection

        right_block_3_out = self.right_GAT_3(right_block_2_out,input.edge_index)
        right_block_3_out = self.elu(self.right_BN3(torch.add(right_block_3_out, right_block_2_out)))

        right_block_4_out = self.right_GAT_4(right_block_3_out,input.edge_index)
        right_block_4_out = self.elu(self.right_BN4(torch.add(right_block_4_out, right_block_3_out)))

        combined = torch.cat((left_block_4_out,right_block_4_out,left_block_3_out,right_block_3_out,left_block_2_out,right_block_2_out,left_block_1_out,right_block_1_out,x), dim=-1)

        combined = self.post_BN(combined)

        x = self.elu(self.postprocess1(combined))
        x = self.elu(self.postprocess2(x))
        x = self.elu(self.postprocess3(x))
        x = self.elu(self.postprocess4(x))
        x = self.elu(self.postprocess5(x))
        x = self.postprocess6(x)

	# Regression Head for Reconstruction Loss
        rcon_output = torch.cat((left_block_4_out, right_block_4_out), dim=-1)
        rcon_output = self.elu(self.rcon1(rcon_output))
        rcon_output = self.elu(self.rcon2(rcon_output))
        rcon_output = self.rcon3(rcon_output)

        return x, rcon_output

class Hybrid_Add_Block(nn.Module):
    def __init__(self, input_dim, output_dim, GAT_heads, edge_dim, MLP_dim, drop_prob=.01, GAT_aggr="mean", GIN_aggr="add"):
        # No need for bias in GAT Convs due to batch norms
        super(Hybrid_Add_Block, self).__init__()
        self.GAT = GATv2Conv(input_dim, int(output_dim/GAT_heads), heads=GAT_heads, edge_dim=edge_dim, bias=False, dropout=drop_prob, aggr=GAT_aggr)
        self.GIN = GINConv(MLP(3, input_dim, MLP_dim, output_dim), aggr=GIN_aggr)
        self.BN1 = BatchNorm(output_dim, track_running_stats=False)
        self.BN2 = BatchNorm(output_dim, track_running_stats=False)
        self.elu = torch.nn.ELU()

    def forward(self, x, edge_index, edge_attr):
        GAT_out = self.GAT(x, edge_index, edge_attr)
        GIN_out = self.GIN(x, edge_index)
        add_out = self.BN1(torch.add(GAT_out, GIN_out))
        block_out = self.elu(self.BN2(torch.add(add_out, x)))

        return block_out
        

class Hybrid_Add_Model(nn.Module):
    def __init__(self, input_dim, output_dim=2, drop_prob=0.1, GAT_aggr="mean", GIN_aggr="add", noise_variance=0.02):
        self.noise_variance = noise_variance
        # No need for bias in GAT Convs due to batch norms
        super(Hybrid_Add_Model, self).__init__()
        self.BN0 = BatchNorm(input_dim, track_running_stats=False, affine=False)

        self.preprocess1 = nn.Linear(input_dim, 72, bias=False)
        self.pre_BN1 = BatchNorm(72, track_running_stats=False)
        self.preprocess2 = nn.Linear(72, 64, bias=False)
        self.pre_BN2 = BatchNorm(64, track_running_stats=False)

        self.block1 = Hybrid_Add_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block2 = Hybrid_Add_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block3 = Hybrid_Add_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block4 = Hybrid_Add_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)

        self.post_BN = BatchNorm(320, track_running_stats=False)
        self.postprocess1 = nn.Linear(320, 256)
        self.postprocess2 = nn.Linear(256, 128)
        self.postprocess3 = nn.Linear(128, 64)
        self.postprocess4 = nn.Linear(64, 32)
        self.postprocess5 = nn.Linear(32, 16)
        self.postprocess6 = nn.Linear(16, output_dim)
        
        self.elu = torch.nn.ELU()
        self.softmax = torch.nn.Softmax(dim=0)

	# Regression head for reconstruction
        self.rcon1 = nn.Linear(64, 64)
        self.rcon2 = nn.Linear(64, 64)
        self.rcon3 = nn.Linear(64, input_dim)

    def forward(self, input):
        x = input.x
        if self.training:
            x += (x.std(dim=0)*self.noise_variance)*torch.randn_like(x)

        x = self.BN0(x)

        x = self.pre_BN1(self.preprocess1(x))
        x = self.elu(x)
        x = self.pre_BN2(self.preprocess2(x))
        x = self.elu(x)

        block1_out = self.block1(x, input.edge_index, input.edge_attr)
        block2_out = self.block2(block1_out, input.edge_index, input.edge_attr)
        block3_out = self.block3(block2_out, input.edge_index, input.edge_attr)
        block4_out = self.block4(block3_out, input.edge_index, input.edge_attr)

        combined = torch.cat((block1_out, block2_out, block3_out, block4_out,x), dim=-1)

        combined = self.post_BN(combined)

        x = self.elu(self.postprocess1(combined))
        x = self.elu(self.postprocess2(x))
        x = self.elu(self.postprocess3(x))
        x = self.elu(self.postprocess4(x))
        x = self.elu(self.postprocess5(x))
        x = self.postprocess6(x)

	# Regression Head for Reconstruction Loss
        rcon_output = self.elu(self.rcon1(block4_out))
        rcon_output = self.elu(self.rcon2(rcon_output))
        rcon_output = self.rcon3(rcon_output)

        return x, rcon_output


class Hybrid_Cat_Block(nn.Module):
    def __init__(self, input_dim, output_dim, GAT_heads, edge_dim, MLP_dim, drop_prob=.01, GAT_aggr="mean", GIN_aggr="add"):
        # No need for bias in GAT Convs due to batch norms
        super(Hybrid_Cat_Block, self).__init__()
        GNN_out_dim = int(output_dim/2)
        self.GAT = GATv2Conv(input_dim, int(GNN_out_dim/GAT_heads), heads=GAT_heads, edge_dim=edge_dim, bias=False, dropout=drop_prob, aggr=GAT_aggr)
        self.GIN = GINConv(MLP(3, input_dim, MLP_dim, GNN_out_dim), aggr=GIN_aggr)
        self.BN1 = BatchNorm(output_dim, track_running_stats=False)
        self.BN2 = BatchNorm(output_dim, track_running_stats=False)
        self.elu = torch.nn.ELU()

    def forward(self, x, edge_index, edge_attr):
        GAT_out = self.GAT(x, edge_index, edge_attr)
        GIN_out = self.GIN(x, edge_index)
        cat_out = self.BN1(torch.cat((GAT_out, GIN_out), dim=-1))
        block_out = self.elu(self.BN2(torch.add(cat_out, x)))

        return block_out
        

class Hybrid_Cat_Model(nn.Module):
    def __init__(self, input_dim, output_dim=2, drop_prob=0.1, GAT_aggr="mean", GIN_aggr="add", noise_variance=0.02):
        self.noise_variance = noise_variance
        # No need for bias in GAT Convs due to batch norms
        super(Hybrid_Cat_Model, self).__init__()
        self.BN0 = BatchNorm(input_dim, track_running_stats=False, affine=False)

        self.preprocess1 = nn.Linear(input_dim, 72, bias=False)
        self.pre_BN1 = BatchNorm(72, track_running_stats=False)
        self.preprocess2 = nn.Linear(72, 64, bias=False)
        self.pre_BN2 = BatchNorm(64, track_running_stats=False)

        self.block1 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block2 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block3 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block4 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)

        self.post_BN = BatchNorm(320, track_running_stats=False)
        self.postprocess1 = nn.Linear(320, 256)
        self.postprocess2 = nn.Linear(256, 128)
        self.postprocess3 = nn.Linear(128, 64)
        self.postprocess4 = nn.Linear(64, 32)
        self.postprocess5 = nn.Linear(32, 16)
        self.postprocess6 = nn.Linear(16, output_dim)
        
        self.elu = torch.nn.ELU()
        self.softmax = torch.nn.Softmax(dim=0)

	# Regression head for reconstruction
        self.rcon1 = nn.Linear(64, 64)
        self.rcon2 = nn.Linear(64, 64)
        self.rcon3 = nn.Linear(64, input_dim)

    def forward(self, input):
        x = input.x
        if self.training:
            x += (x.std(dim=0)*self.noise_variance)*torch.randn_like(x)

        x = self.BN0(x)

        x = self.pre_BN1(self.preprocess1(x))
        x = self.elu(x)
        x = self.pre_BN2(self.preprocess2(x))
        x = self.elu(x)

        block1_out = self.block1(x, input.edge_index, input.edge_attr)
        block2_out = self.block2(block1_out, input.edge_index, input.edge_attr)
        block3_out = self.block3(block2_out, input.edge_index, input.edge_attr)
        block4_out = self.block4(block3_out, input.edge_index, input.edge_attr)

        combined = torch.cat((block1_out, block2_out, block3_out, block4_out,x), dim=-1)

        combined = self.post_BN(combined)

        x = self.elu(self.postprocess1(combined))
        x = self.elu(self.postprocess2(x))
        x = self.elu(self.postprocess3(x))
        x = self.elu(self.postprocess4(x))
        x = self.elu(self.postprocess5(x))
        x = self.postprocess6(x)

	# Regression Head for Reconstruction Loss
        rcon_output = self.elu(self.rcon1(block4_out))
        rcon_output = self.elu(self.rcon2(rcon_output))
        rcon_output = self.rcon3(rcon_output)

        return x, rcon_output

class Hybrid_1g8(nn.Module):
    def __init__(self, input_dim, output_dim=2, drop_prob=0.1, GAT_aggr="mean", GIN_aggr="add", noise_variance=0.02):
        self.noise_variance = noise_variance
        # No need for bias in GAT Convs due to batch norms
        super(Hybrid_1g8, self).__init__()
        self.BN0 = BatchNorm(input_dim, track_running_stats=False, affine=False)

        self.preprocess1 = nn.Linear(input_dim, 72, bias=False)
        self.pre_BN1 = BatchNorm(72, track_running_stats=False)
        self.preprocess2 = nn.Linear(72, 64, bias=False)
        self.pre_BN2 = BatchNorm(64, track_running_stats=False)

        self.block1 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block2 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block3 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block4 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block5 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block6 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block7 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block8 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)

        self.post_BN = BatchNorm(576, track_running_stats=False)
        self.postprocess1 = nn.Linear(576, 256)
        self.postprocess2 = nn.Linear(256, 128)
        self.postprocess3 = nn.Linear(128, 64)
        self.postprocess4 = nn.Linear(64, 32)
        self.postprocess5 = nn.Linear(32, 16)
        self.postprocess6 = nn.Linear(16, output_dim)
        
        self.elu = torch.nn.ELU()
        self.softmax = torch.nn.Softmax(dim=0)

	# Regression head for reconstruction
        self.rcon1 = nn.Linear(64, 64)
        self.rcon2 = nn.Linear(64, 64)
        self.rcon3 = nn.Linear(64, input_dim)

    def forward(self, input):
        x = input.x
        if self.training:
            x += (x.std(dim=0)*self.noise_variance)*torch.randn_like(x)

        x = self.BN0(x)

        x = self.pre_BN1(self.preprocess1(x))
        x = self.elu(x)
        x = self.pre_BN2(self.preprocess2(x))
        x = self.elu(x)

        block1_out = self.block1(x, input.edge_index, input.edge_attr)
        block2_out = self.block2(block1_out, input.edge_index, input.edge_attr)
        block3_out = self.block3(block2_out, input.edge_index, input.edge_attr)
        block4_out = self.block4(block3_out, input.edge_index, input.edge_attr)
        block5_out = self.block5(block4_out, input.edge_index, input.edge_attr)
        block6_out = self.block6(block5_out, input.edge_index, input.edge_attr)
        block7_out = self.block7(block6_out, input.edge_index, input.edge_attr)
        block8_out = self.block8(block7_out, input.edge_index, input.edge_attr)

        combined = torch.cat((block1_out, block2_out, block3_out, block4_out,
         block5_out, block6_out, block7_out, block8_out, x), dim=-1)

        combined = self.post_BN(combined)

        x = self.elu(self.postprocess1(combined))
        x = self.elu(self.postprocess2(x))
        x = self.elu(self.postprocess3(x))
        x = self.elu(self.postprocess4(x))
        x = self.elu(self.postprocess5(x))
        x = self.postprocess6(x)

	# Regression Head for Reconstruction Loss
        rcon_output = self.elu(self.rcon1(block8_out))
        rcon_output = self.elu(self.rcon2(rcon_output))
        rcon_output = self.rcon3(rcon_output)

        return x, rcon_output
    
class Hybrid_1g8_noisy(nn.Module):
    def __init__(self, input_dim, output_dim=2, drop_prob=0.1, GAT_aggr="mean", GIN_aggr="add", node_noise_variance=0.02, edge_noise_variance=0.02):
        self.node_noise_variance = node_noise_variance
        self.edge_noise_variance = edge_noise_variance
        # No need for bias in GAT Convs due to batch norms
        super(Hybrid_1g8_noisy, self).__init__()
        self.BN0 = BatchNorm(input_dim, track_running_stats=False, affine=False)

        self.preprocess1 = nn.Linear(input_dim, 72, bias=False)
        self.pre_BN1 = BatchNorm(72, track_running_stats=False)
        self.preprocess2 = nn.Linear(72, 64, bias=False)
        self.pre_BN2 = BatchNorm(64, track_running_stats=False)

        self.block1 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block2 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block3 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block4 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block5 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block6 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block7 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block8 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)

        self.post_BN = BatchNorm(576, track_running_stats=False)
        self.postprocess1 = nn.Linear(576, 256)
        self.postprocess2 = nn.Linear(256, 128)
        self.postprocess3 = nn.Linear(128, 64)
        self.postprocess4 = nn.Linear(64, 32)
        self.postprocess5 = nn.Linear(32, 16)
        self.postprocess6 = nn.Linear(16, output_dim)
        
        self.elu = torch.nn.ELU()
        self.softmax = torch.nn.Softmax(dim=0)

	# Regression head for reconstruction
        self.rcon1 = nn.Linear(64, 64)
        self.rcon2 = nn.Linear(64, 64)
        self.rcon3 = nn.Linear(64, input_dim)

    def forward(self, input):
        x =         input.x
        edge_attr = input.edge_attr
        if self.training:
            x += (x.std(dim=0)*self.node_noise_variance)*torch.randn_like(x)
            edge_attr += (edge_attr.std(dim=0)*self.edge_noise_variance)*torch.randn_like(edge_attr)
            '''
            This is maybe a little odd, unlike the features, which get renormalized after adding noise, I'm not doing that for the edges.
            My thought process here is that this is more akin to 'flipping' one hot encodings in aggregate, than renormalizing them would be.add()
            '''
            

        x = self.BN0(x)

        x = self.pre_BN1(self.preprocess1(x))
        x = self.elu(x)
        x = self.pre_BN2(self.preprocess2(x))
        x = self.elu(x)

        block1_out = self.block1(x, input.edge_index, edge_attr)
        block2_out = self.block2(block1_out, input.edge_index, edge_attr)
        block3_out = self.block3(block2_out, input.edge_index, edge_attr)
        block4_out = self.block4(block3_out, input.edge_index, edge_attr)
        block5_out = self.block5(block4_out, input.edge_index, edge_attr)
        block6_out = self.block6(block5_out, input.edge_index, edge_attr)
        block7_out = self.block7(block6_out, input.edge_index, edge_attr)
        block8_out = self.block8(block7_out, input.edge_index, edge_attr)

        combined = torch.cat((block1_out, block2_out, block3_out, block4_out,
         block5_out, block6_out, block7_out, block8_out, x), dim=-1)

        combined = self.post_BN(combined)

        x = self.elu(self.postprocess1(combined))
        x = self.elu(self.postprocess2(x))
        x = self.elu(self.postprocess3(x))
        x = self.elu(self.postprocess4(x))
        x = self.elu(self.postprocess5(x))
        x = self.postprocess6(x)

	# Regression Head for Reconstruction Loss
        rcon_output = self.elu(self.rcon1(block8_out))
        rcon_output = self.elu(self.rcon2(rcon_output))
        rcon_output = self.rcon3(rcon_output)

        return x, rcon_output

class Hybrid_1g12(nn.Module):
    def __init__(self, input_dim, output_dim=2, drop_prob=0.1, GAT_aggr="mean", GIN_aggr="add", noise_variance=0.02):
        self.noise_variance = noise_variance
        # No need for bias in GAT Convs due to batch norms
        super(Hybrid_1g12, self).__init__()
        self.BN0 = BatchNorm(input_dim, track_running_stats=False, affine=False)

        self.preprocess1 = nn.Linear(input_dim, 72, bias=False)
        self.pre_BN1 = BatchNorm(72, track_running_stats=False)
        self.preprocess2 = nn.Linear(72, 64, bias=False)
        self.pre_BN2 = BatchNorm(64, track_running_stats=False)

        self.block1 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block2 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block3 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block4 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block5 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block6 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block7 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block8 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block9 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block10 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block11 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block12 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)

        self.post_BN = BatchNorm(832, track_running_stats=False)
        self.postprocess1 = nn.Linear(832, 256)
        self.postprocess2 = nn.Linear(256, 128)
        self.postprocess3 = nn.Linear(128, 64)
        self.postprocess4 = nn.Linear(64, 32)
        self.postprocess5 = nn.Linear(32, 16)
        self.postprocess6 = nn.Linear(16, output_dim)
        
        self.elu = torch.nn.ELU()
        self.softmax = torch.nn.Softmax(dim=0)

	# Regression head for reconstruction
        self.rcon1 = nn.Linear(64, 64)
        self.rcon2 = nn.Linear(64, 64)
        self.rcon3 = nn.Linear(64, input_dim)

    def forward(self, input):
        x = input.x
        if self.training:
            x += (x.std(dim=0)*self.noise_variance)*torch.randn_like(x)

        x = self.BN0(x)

        x = self.pre_BN1(self.preprocess1(x))
        x = self.elu(x)
        x = self.pre_BN2(self.preprocess2(x))
        x = self.elu(x)

        block1_out = self.block1(x, input.edge_index, input.edge_attr)
        block2_out = self.block2(block1_out, input.edge_index, input.edge_attr)
        block3_out = self.block3(block2_out, input.edge_index, input.edge_attr)
        block4_out = self.block4(block3_out, input.edge_index, input.edge_attr)
        block5_out = self.block5(block4_out, input.edge_index, input.edge_attr)
        block6_out = self.block6(block5_out, input.edge_index, input.edge_attr)
        block7_out = self.block7(block6_out, input.edge_index, input.edge_attr)
        block8_out = self.block8(block7_out, input.edge_index, input.edge_attr)
        block9_out = self.block9(block8_out, input.edge_index, input.edge_attr)
        block10_out = self.block10(block9_out, input.edge_index, input.edge_attr)
        block11_out = self.block11(block10_out, input.edge_index, input.edge_attr)
        block12_out = self.block12(block11_out, input.edge_index, input.edge_attr)

        combined = torch.cat((block1_out, block2_out, block3_out, block4_out,
         block5_out, block6_out, block7_out, block8_out, block9_out, block10_out,
          block11_out, block12_out, x), dim=-1)

        combined = self.post_BN(combined)

        x = self.elu(self.postprocess1(combined))
        x = self.elu(self.postprocess2(x))
        x = self.elu(self.postprocess3(x))
        x = self.elu(self.postprocess4(x))
        x = self.elu(self.postprocess5(x))
        x = self.postprocess6(x)

	# Regression Head for Reconstruction Loss
        rcon_output = self.elu(self.rcon1(block12_out))
        rcon_output = self.elu(self.rcon2(rcon_output))
        rcon_output = self.rcon3(rcon_output)

        return x, rcon_output

class Hybrid_2g4(nn.Module):
    def __init__(self, input_dim, output_dim=2, drop_prob=0.1, GAT_aggr="mean", GIN_aggr="add", noise_variance=0.02):
        self.noise_variance = noise_variance
        # No need for bias in GAT Convs due to batch norms
        super(Hybrid_2g4, self).__init__()
        self.BN0 = BatchNorm(input_dim, track_running_stats=False, affine=False)

        self.preprocess1 = nn.Linear(input_dim, 72, bias=False)
        self.pre_BN1 = BatchNorm(72, track_running_stats=False)
        self.preprocess2 = nn.Linear(72, 64, bias=False)
        self.pre_BN2 = BatchNorm(64, track_running_stats=False)

        self.block1 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block2 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block3 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block4 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)

        self.post_BN = BatchNorm(576, track_running_stats=False)
        self.postprocess1 = nn.Linear(576, 256)
        self.postprocess2 = nn.Linear(256, 128)
        self.postprocess3 = nn.Linear(128, 64)
        self.postprocess4 = nn.Linear(64, 32)
        self.postprocess5 = nn.Linear(32, 16)
        self.postprocess6 = nn.Linear(16, output_dim)
        
        self.elu = torch.nn.ELU()
        self.softmax = torch.nn.Softmax(dim=0)

	# Regression head for reconstruction
        self.rcon1 = nn.Linear(64, 64)
        self.rcon2 = nn.Linear(64, 64)
        self.rcon3 = nn.Linear(64, input_dim)

    def forward(self, input):
        x = input.x
        if self.training:
            x += (x.std(dim=0)*self.noise_variance)*torch.randn_like(x)

        x = self.BN0(x)

        x = self.pre_BN1(self.preprocess1(x))
        x = self.elu(x)
        x = self.pre_BN2(self.preprocess2(x))
        x = self.elu(x)

        # Group 1
        block1_out = self.block1(x, input.edge_index, input.edge_attr)
        block2_out = self.block2(block1_out, input.edge_index, input.edge_attr)
        block3_out = self.block3(block2_out, input.edge_index, input.edge_attr)
        block4_out = self.block4(block3_out, input.edge_index, input.edge_attr)
        # Group 2
        block5_out = self.block1(block4_out, input.edge_index, input.edge_attr)
        block6_out = self.block2(block5_out, input.edge_index, input.edge_attr)
        block7_out = self.block3(block6_out, input.edge_index, input.edge_attr)
        block8_out = self.block4(block7_out, input.edge_index, input.edge_attr)

        combined = torch.cat((block1_out, block2_out, block3_out, block4_out,
         block5_out, block6_out, block7_out, block8_out, x), dim=-1)

        combined = self.post_BN(combined)

        x = self.elu(self.postprocess1(combined))
        x = self.elu(self.postprocess2(x))
        x = self.elu(self.postprocess3(x))
        x = self.elu(self.postprocess4(x))
        x = self.elu(self.postprocess5(x))
        x = self.postprocess6(x)

	# Regression Head for Reconstruction Loss
        rcon_output = self.elu(self.rcon1(block8_out))
        rcon_output = self.elu(self.rcon2(rcon_output))
        rcon_output = self.rcon3(rcon_output)

        return x, rcon_output

class Hybrid_4g2(nn.Module):
    def __init__(self, input_dim, output_dim=2, drop_prob=0.1, GAT_aggr="mean", GIN_aggr="add", noise_variance=0.02):
        self.noise_variance = noise_variance
        # No need for bias in GAT Convs due to batch norms
        super(Hybrid_4g2, self).__init__()
        self.BN0 = BatchNorm(input_dim, track_running_stats=False, affine=False)

        self.preprocess1 = nn.Linear(input_dim, 72, bias=False)
        self.pre_BN1 = BatchNorm(72, track_running_stats=False)
        self.preprocess2 = nn.Linear(72, 64, bias=False)
        self.pre_BN2 = BatchNorm(64, track_running_stats=False)

        self.block1 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.block2 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        
        self.post_BN = BatchNorm(576, track_running_stats=False)
        self.postprocess1 = nn.Linear(576, 256)
        self.postprocess2 = nn.Linear(256, 128)
        self.postprocess3 = nn.Linear(128, 64)
        self.postprocess4 = nn.Linear(64, 32)
        self.postprocess5 = nn.Linear(32, 16)
        self.postprocess6 = nn.Linear(16, output_dim)
        
        self.elu = torch.nn.ELU()
        self.softmax = torch.nn.Softmax(dim=0)

	# Regression head for reconstruction
        self.rcon1 = nn.Linear(64, 64)
        self.rcon2 = nn.Linear(64, 64)
        self.rcon3 = nn.Linear(64, input_dim)

    def forward(self, input):
        x = input.x
        if self.training:
            x += (x.std(dim=0)*self.noise_variance)*torch.randn_like(x)

        x = self.BN0(x)

        x = self.pre_BN1(self.preprocess1(x))
        x = self.elu(x)
        x = self.pre_BN2(self.preprocess2(x))
        x = self.elu(x)

        # Group 1
        block1_out = self.block1(x, input.edge_index, input.edge_attr)
        block2_out = self.block2(block1_out, input.edge_index, input.edge_attr)
        # Group 2
        block3_out = self.block1(block2_out, input.edge_index, input.edge_attr)
        block4_out = self.block2(block3_out, input.edge_index, input.edge_attr)
        # Group 3
        block5_out = self.block1(block4_out, input.edge_index, input.edge_attr)
        block6_out = self.block2(block5_out, input.edge_index, input.edge_attr)
        # Group 4
        block7_out = self.block1(block6_out, input.edge_index, input.edge_attr)
        block8_out = self.block2(block7_out, input.edge_index, input.edge_attr)

        combined = torch.cat((block1_out, block2_out, block3_out, block4_out,
         block5_out, block6_out, block7_out, block8_out, x), dim=-1)

        combined = self.post_BN(combined)

        x = self.elu(self.postprocess1(combined))
        x = self.elu(self.postprocess2(x))
        x = self.elu(self.postprocess3(x))
        x = self.elu(self.postprocess4(x))
        x = self.elu(self.postprocess5(x))
        x = self.postprocess6(x)

	# Regression Head for Reconstruction Loss
        rcon_output = self.elu(self.rcon1(block8_out))
        rcon_output = self.elu(self.rcon2(rcon_output))
        rcon_output = self.rcon3(rcon_output)

        return x, rcon_output

class Hybrid_8g1(nn.Module):
    def __init__(self, input_dim, output_dim=2, drop_prob=0.1, GAT_aggr="mean", GIN_aggr="add", noise_variance=0.02):
        self.noise_variance = noise_variance
        # No need for bias in GAT Convs due to batch norms
        super(Hybrid_8g1, self).__init__()
        self.BN0 = BatchNorm(input_dim, track_running_stats=False, affine=False)

        self.preprocess1 = nn.Linear(input_dim, 72, bias=False)
        self.pre_BN1 = BatchNorm(72, track_running_stats=False)
        self.preprocess2 = nn.Linear(72, 64, bias=False)
        self.pre_BN2 = BatchNorm(64, track_running_stats=False)

        self.block1 = Hybrid_Cat_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        
        self.post_BN = BatchNorm(576, track_running_stats=False)
        self.postprocess1 = nn.Linear(576, 256)
        self.postprocess2 = nn.Linear(256, 128)
        self.postprocess3 = nn.Linear(128, 64)
        self.postprocess4 = nn.Linear(64, 32)
        self.postprocess5 = nn.Linear(32, 16)
        self.postprocess6 = nn.Linear(16, output_dim)
        
        self.elu = torch.nn.ELU()
        self.softmax = torch.nn.Softmax(dim=0)

	# Regression head for reconstruction
        self.rcon1 = nn.Linear(64, 64)
        self.rcon2 = nn.Linear(64, 64)
        self.rcon3 = nn.Linear(64, input_dim)

    def forward(self, input):
        x = input.x
        if self.training:
            x += (x.std(dim=0)*self.noise_variance)*torch.randn_like(x)

        x = self.BN0(x)

        x = self.pre_BN1(self.preprocess1(x))
        x = self.elu(x)
        x = self.pre_BN2(self.preprocess2(x))
        x = self.elu(x)

        block1_out = self.block1(x, input.edge_index, input.edge_attr)
        block2_out = self.block1(block1_out, input.edge_index, input.edge_attr)
        block3_out = self.block1(block2_out, input.edge_index, input.edge_attr)
        block4_out = self.block1(block3_out, input.edge_index, input.edge_attr)
        block5_out = self.block1(block4_out, input.edge_index, input.edge_attr)
        block6_out = self.block1(block5_out, input.edge_index, input.edge_attr)
        block7_out = self.block1(block6_out, input.edge_index, input.edge_attr)
        block8_out = self.block1(block7_out, input.edge_index, input.edge_attr)

        combined = torch.cat((block1_out, block2_out, block3_out, block4_out,
         block5_out, block6_out, block7_out, block8_out, x), dim=-1)

        combined = self.post_BN(combined)

        x = self.elu(self.postprocess1(combined))
        x = self.elu(self.postprocess2(x))
        x = self.elu(self.postprocess3(x))
        x = self.elu(self.postprocess4(x))
        x = self.elu(self.postprocess5(x))
        x = self.postprocess6(x)

	# Regression Head for Reconstruction Loss
        rcon_output = self.elu(self.rcon1(block8_out))
        rcon_output = self.elu(self.rcon2(rcon_output))
        rcon_output = self.rcon3(rcon_output)

        return x, rcon_output

class Hybrid_Block(nn.Module):
    def __init__(self, input_dim, output_dim, GAT_heads, edge_dim, MLP_dim, drop_prob=.01, GAT_aggr="mean", GIN_aggr="add"):
        # No need for bias in GAT Convs due to batch norms
        super(Hybrid_Block, self).__init__()
        GNN_out_dim = int(output_dim/2)
        self.GAT = GATv2Conv(input_dim, int(GNN_out_dim/GAT_heads), heads=GAT_heads, edge_dim=edge_dim, bias=False, dropout=drop_prob, aggr=GAT_aggr)
        self.GIN = GINConv(MLP(3, input_dim, MLP_dim, GNN_out_dim), aggr=GIN_aggr)
        self.BN = BatchNorm(output_dim, track_running_stats=False)

    def forward(self, x, edge_index, edge_attr):
        GAT_out = self.GAT(x, edge_index, edge_attr)
        GIN_out = self.GIN(x, edge_index)
        block_out = self.BN(torch.cat((GAT_out, GIN_out), dim=-1))
    
        return block_out
        

class Hybrid_GRU(nn.Module):
    def __init__(self, input_dim, output_dim=2, drop_prob=0.1, GAT_aggr="mean", GIN_aggr="add", noise_variance=0.02):
        self.noise_variance = noise_variance
        # No need for bias in GAT Convs due to batch norms
        super(Hybrid_GRU, self).__init__()
        self.BN0 = BatchNorm(input_dim, track_running_stats=False, affine=False)

        self.preprocess1 = nn.Linear(input_dim, 72, bias=False)
        self.pre_BN1 = BatchNorm(72, track_running_stats=False)
        self.preprocess2 = nn.Linear(72, 64, bias=False)
        self.pre_BN2 = BatchNorm(64, track_running_stats=False)
        self.pre_BN3 = BatchNorm(64, track_running_stats=False)

        self.block1 = Hybrid_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.BN1 = BatchNorm(64, track_running_stats=False)
        self.block2 = Hybrid_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.BN2 = BatchNorm(64, track_running_stats=False)
        self.block3 = Hybrid_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.BN3 = BatchNorm(64, track_running_stats=False)
        self.block4 = Hybrid_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.BN4 = BatchNorm(64, track_running_stats=False)
        self.block5 = Hybrid_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.BN5 = BatchNorm(64, track_running_stats=False)
        self.block6 = Hybrid_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.BN6 = BatchNorm(64, track_running_stats=False)
        self.block7 = Hybrid_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.BN7 = BatchNorm(64, track_running_stats=False)
        self.block8 = Hybrid_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.BN8 = BatchNorm(64, track_running_stats=False)

        self.post_BN = BatchNorm(576, track_running_stats=False)
        self.postprocess1 = nn.Linear(576, 256)
        self.postprocess2 = nn.Linear(256, 128)
        self.postprocess3 = nn.Linear(128, 64)
        self.postprocess4 = nn.Linear(64, 32)
        self.postprocess5 = nn.Linear(32, 16)
        self.postprocess6 = nn.Linear(16, output_dim)
        
        self.GRU = torch.nn.GRUCell(64,64)
        self.elu = torch.nn.ELU()
        self.softmax = torch.nn.Softmax(dim=0)

	# Regression head for reconstruction
        self.rcon1 = nn.Linear(64, 64)
        self.rcon2 = nn.Linear(64, 64)
        self.rcon3 = nn.Linear(64, input_dim)

    def forward(self, input):
        x = input.x
        if self.training:
            x += (x.std(dim=0)*self.noise_variance)*torch.randn_like(x)

        x = self.BN0(x)

        x = self.pre_BN1(self.preprocess1(x))
        x = self.elu(x)
        x = self.pre_BN2(self.preprocess2(x))
        x = self.elu(x)
        x  = self.GRU(x)
        x = self.elu(self.pre_BN3(x))

        block1_out = self.block1(x, input.edge_index, input.edge_attr)
        block1_out = self.GRU(block1_out, x)
        block1_out = self.elu(self.BN1(block1_out))

        block2_out = self.block2(block1_out, input.edge_index, input.edge_attr)
        block2_out = self.GRU(block2_out, block1_out)
        block2_out = self.elu(self.BN2(block2_out))

        block3_out = self.block3(block2_out, input.edge_index, input.edge_attr)
        block3_out = self.GRU(block3_out, block2_out)
        block3_out = self.elu(self.BN3(block3_out))

        block4_out = self.block4(block3_out, input.edge_index, input.edge_attr)
        block4_out = self.GRU(block4_out, block3_out)
        block4_out = self.elu(self.BN4(block4_out))

        block5_out = self.block5(block4_out, input.edge_index, input.edge_attr)
        block5_out = self.GRU(block5_out, block4_out)
        block5_out = self.elu(self.BN5(block5_out))

        block6_out = self.block6(block5_out, input.edge_index, input.edge_attr)
        block6_out = self.GRU(block6_out, block5_out)
        block6_out = self.elu(self.BN6(block6_out))

        block7_out = self.block7(block6_out, input.edge_index, input.edge_attr)
        block7_out = self.GRU(block7_out, block6_out)
        block7_out = self.elu(self.BN7(block7_out))

        block8_out = self.block8(block7_out, input.edge_index, input.edge_attr)
        block8_out = self.GRU(block8_out, block7_out)
        block8_out = self.elu(self.BN8(block8_out))

        combined = torch.cat((block1_out, block2_out, block3_out, block4_out,
         block5_out, block6_out, block7_out, block8_out, x), dim=-1)

        combined = self.post_BN(combined)

        x = self.elu(self.postprocess1(combined))
        x = self.elu(self.postprocess2(x))
        x = self.elu(self.postprocess3(x))
        x = self.elu(self.postprocess4(x))
        x = self.elu(self.postprocess5(x))
        x = self.postprocess6(x)

	# Regression Head for Reconstruction Loss
        rcon_output = self.elu(self.rcon1(block8_out))
        rcon_output = self.elu(self.rcon2(rcon_output))
        rcon_output = self.rcon3(rcon_output)

        return x, rcon_output

class Hybrid_GRU_Unique(nn.Module):
    def __init__(self, input_dim, output_dim=2, drop_prob=0.1, GAT_aggr="mean", GIN_aggr="add", noise_variance=0.02):
        self.noise_variance = noise_variance
        # No need for bias in GAT Convs due to batch norms
        super(Hybrid_GRU_Unique, self).__init__()
        self.BN0 = BatchNorm(input_dim, track_running_stats=False, affine=False)

        self.preprocess1 = nn.Linear(input_dim, 72, bias=False)
        self.pre_BN1 = BatchNorm(72, track_running_stats=False)
        self.preprocess2 = nn.Linear(72, 64, bias=False)
        self.pre_BN2 = BatchNorm(64, track_running_stats=False)
        self.GRU0 = torch.nn.GRUCell(64,64)
        self.pre_BN3 = BatchNorm(64, track_running_stats=False)

        self.block1 = Hybrid_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.GRU1 = torch.nn.GRUCell(64,64)
        self.BN1 = BatchNorm(64, track_running_stats=False)
        
        self.block2 = Hybrid_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.GRU2 = torch.nn.GRUCell(64,64)
        self.BN2 = BatchNorm(64, track_running_stats=False)
        
        self.block3 = Hybrid_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.GRU3 = torch.nn.GRUCell(64,64)
        self.BN3 = BatchNorm(64, track_running_stats=False)
        
        self.block4 = Hybrid_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.GRU4 = torch.nn.GRUCell(64,64)
        self.BN4 = BatchNorm(64, track_running_stats=False)

        self.block5 = Hybrid_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.GRU5 = torch.nn.GRUCell(64,64)
        self.BN5 = BatchNorm(64, track_running_stats=False)

        self.block6 = Hybrid_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.GRU6 = torch.nn.GRUCell(64,64)
        self.BN6 = BatchNorm(64, track_running_stats=False)

        self.block7 = Hybrid_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.GRU7 = torch.nn.GRUCell(64,64)
        self.BN7 = BatchNorm(64, track_running_stats=False)

        self.block8 = Hybrid_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.GRU8 = torch.nn.GRUCell(64,64)
        self.BN8 = BatchNorm(64, track_running_stats=False)

        self.post_BN = BatchNorm(576, track_running_stats=False)
        self.postprocess1 = nn.Linear(576, 256)
        self.postprocess2 = nn.Linear(256, 128)
        self.postprocess3 = nn.Linear(128, 64)
        self.postprocess4 = nn.Linear(64, 32)
        self.postprocess5 = nn.Linear(32, 16)
        self.postprocess6 = nn.Linear(16, output_dim)
        
        self.elu = torch.nn.ELU()
        self.softmax = torch.nn.Softmax(dim=0)

	# Regression head for reconstruction
        self.rcon1 = nn.Linear(64, 64)
        self.rcon2 = nn.Linear(64, 64)
        self.rcon3 = nn.Linear(64, input_dim)

    def forward(self, input):
        x = input.x
        if self.training:
            x += (x.std(dim=0)*self.noise_variance)*torch.randn_like(x)

        x = self.BN0(x)

        x = self.pre_BN1(self.preprocess1(x))
        x = self.elu(x)
        x = self.pre_BN2(self.preprocess2(x))
        x = self.elu(x)
        x = self.GRU0(x)
        x = self.elu(self.pre_BN3(x))

        block1_out = self.block1(x, input.edge_index, input.edge_attr)
        block1_out = self.GRU1(block1_out, x)
        block1_out = self.elu(self.BN1(block1_out))

        block2_out = self.block2(block1_out, input.edge_index, input.edge_attr)
        block2_out = self.GRU2(block2_out, block1_out)
        block2_out = self.elu(self.BN2(block2_out))

        block3_out = self.block3(block2_out, input.edge_index, input.edge_attr)
        block3_out = self.GRU3(block3_out, block2_out)
        block3_out = self.elu(self.BN3(block3_out))

        block4_out = self.block4(block3_out, input.edge_index, input.edge_attr)
        block4_out = self.GRU4(block4_out, block3_out)
        block4_out = self.elu(self.BN4(block4_out))

        block5_out = self.block5(block4_out, input.edge_index, input.edge_attr)
        block5_out = self.GRU5(block5_out, block4_out)
        block5_out = self.elu(self.BN5(block5_out))

        block6_out = self.block6(block5_out, input.edge_index, input.edge_attr)
        block6_out = self.GRU6(block6_out, block5_out)
        block6_out = self.elu(self.BN6(block6_out))

        block7_out = self.block7(block6_out, input.edge_index, input.edge_attr)
        block7_out = self.GRU7(block7_out, block6_out)
        block7_out = self.elu(self.BN7(block7_out))

        block8_out = self.block8(block7_out, input.edge_index, input.edge_attr)
        block8_out = self.GRU8(block8_out, block7_out)
        block8_out = self.elu(self.BN8(block8_out))

        combined = torch.cat((block1_out, block2_out, block3_out, block4_out,
         block5_out, block6_out, block7_out, block8_out, x), dim=-1)

        combined = self.post_BN(combined)

        x = self.elu(self.postprocess1(combined))
        x = self.elu(self.postprocess2(x))
        x = self.elu(self.postprocess3(x))
        x = self.elu(self.postprocess4(x))
        x = self.elu(self.postprocess5(x))
        x = self.postprocess6(x)

	# Regression Head for Reconstruction Loss
        rcon_output = self.elu(self.rcon1(block8_out))
        rcon_output = self.elu(self.rcon2(rcon_output))
        rcon_output = self.rcon3(rcon_output)

        return x, rcon_output

class Hybrid_Weighted(nn.Module):
    def __init__(self, input_dim, output_dim=2, drop_prob=0.1, GAT_aggr="mean", GIN_aggr="add", noise_variance=0.02):
        self.noise_variance = noise_variance
        # No need for bias in GAT Convs due to batch norms
        super(Hybrid_Weighted, self).__init__()
        self.BN0 = BatchNorm(input_dim, track_running_stats=False, affine=False)

        self.preprocess1 = nn.Linear(input_dim, 72, bias=False)
        self.pre_BN1 = BatchNorm(72, track_running_stats=False)
        self.preprocess2 = nn.Linear(72, 64, bias=False)
        self.pre_BN2 = BatchNorm(64, track_running_stats=False)
        self.GRU0 = torch.nn.GRUCell(64,64)
        self.pre_BN3 = BatchNorm(64, track_running_stats=False)

        self.block1 = Hybrid_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.BN1 = BatchNorm(64, track_running_stats=False)
        
        self.block2 = Hybrid_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.BN2 = BatchNorm(64, track_running_stats=False)
        
        self.block3 = Hybrid_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.BN3 = BatchNorm(64, track_running_stats=False)
        
        self.block4 = Hybrid_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.BN4 = BatchNorm(64, track_running_stats=False)

        self.block5 = Hybrid_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.BN5 = BatchNorm(64, track_running_stats=False)

        self.block6 = Hybrid_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.BN6 = BatchNorm(64, track_running_stats=False)

        self.block7 = Hybrid_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.BN7 = BatchNorm(64, track_running_stats=False)

        self.block8 = Hybrid_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.BN8 = BatchNorm(64, track_running_stats=False)

        self.post_BN = BatchNorm(576, track_running_stats=False)
        self.postprocess1 = nn.Linear(576, 256)
        self.postprocess2 = nn.Linear(256, 128)
        self.postprocess3 = nn.Linear(128, 64)
        self.postprocess4 = nn.Linear(64, 32)
        self.postprocess5 = nn.Linear(32, 16)
        self.postprocess6 = nn.Linear(16, output_dim)
        
        self.w = torch.nn.Parameter(torch.ones(1))
        self.elu = torch.nn.ELU()
        self.softmax = torch.nn.Softmax(dim=0)

	# Regression head for reconstruction
        self.rcon1 = nn.Linear(64, 64)
        self.rcon2 = nn.Linear(64, 64)
        self.rcon3 = nn.Linear(64, input_dim)

    def forward(self, input):
        x = input.x
        if self.training:
            x += (x.std(dim=0)*self.noise_variance)*torch.randn_like(x)

        x = self.BN0(x)

        x = self.pre_BN1(self.preprocess1(x))
        x = self.elu(x)
        x = self.pre_BN2(self.preprocess2(x))
        x = self.elu(x)
        x = self.GRU0(x)
        x = self.elu(self.pre_BN3(x))

        block1_out = self.block1(x, input.edge_index, input.edge_attr)
        block1_out = torch.add(block1_out * self.w, x)
        block1_out = self.elu(self.BN1(block1_out))

        block2_out = self.block2(block1_out, input.edge_index, input.edge_attr)
        block2_out = torch.add(block2_out * self.w, block1_out)
        block2_out = self.elu(self.BN2(block2_out))

        block3_out = self.block3(block2_out, input.edge_index, input.edge_attr)
        block3_out = torch.add(block3_out * self.w, block2_out)
        block3_out = self.elu(self.BN3(block3_out))

        block4_out = self.block4(block3_out, input.edge_index, input.edge_attr)
        block4_out = torch.add(block4_out * self.w, block3_out)
        block4_out = self.elu(self.BN4(block4_out))

        block5_out = self.block5(block4_out, input.edge_index, input.edge_attr)
        block5_out = torch.add(block5_out * self.w, block4_out)
        block5_out = self.elu(self.BN5(block5_out))

        block6_out = self.block6(block5_out, input.edge_index, input.edge_attr)
        block6_out = torch.add(block6_out * self.w, block5_out)
        block6_out = self.elu(self.BN6(block6_out))

        block7_out = self.block7(block6_out, input.edge_index, input.edge_attr)
        block7_out = torch.add(block7_out * self.w, block6_out)
        block7_out = self.elu(self.BN7(block7_out))

        block8_out = self.block8(block7_out, input.edge_index, input.edge_attr)
        block8_out = torch.add(block8_out * self.w, block7_out)
        block8_out = self.elu(self.BN8(block8_out))

        combined = torch.cat((block1_out, block2_out, block3_out, block4_out,
         block5_out, block6_out, block7_out, block8_out, x), dim=-1)

        combined = self.post_BN(combined)

        x = self.elu(self.postprocess1(combined))
        x = self.elu(self.postprocess2(x))
        x = self.elu(self.postprocess3(x))
        x = self.elu(self.postprocess4(x))
        x = self.elu(self.postprocess5(x))
        x = self.postprocess6(x)

	# Regression Head for Reconstruction Loss
        rcon_output = self.elu(self.rcon1(block8_out))
        rcon_output = self.elu(self.rcon2(rcon_output))
        rcon_output = self.rcon3(rcon_output)

        return x, rcon_output

class Hybrid_Weighted_Unique(nn.Module):
    def __init__(self, input_dim, output_dim=2, drop_prob=0.1, GAT_aggr="mean", GIN_aggr="add", noise_variance=0.02):
        self.noise_variance = noise_variance
        # No need for bias in GAT Convs due to batch norms
        super(Hybrid_Weighted_Unique, self).__init__()
        self.BN0 = BatchNorm(input_dim, track_running_stats=False, affine=False)

        self.preprocess1 = nn.Linear(input_dim, 72, bias=False)
        self.pre_BN1 = BatchNorm(72, track_running_stats=False)
        self.preprocess2 = nn.Linear(72, 64, bias=False)
        self.pre_BN2 = BatchNorm(64, track_running_stats=False)
        self.GRU0 = torch.nn.GRUCell(64,64)
        self.pre_BN3 = BatchNorm(64, track_running_stats=False)

        self.block1 = Hybrid_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.w1 = torch.nn.Parameter(torch.ones(1))
        self.BN1 = BatchNorm(64, track_running_stats=False)
        
        self.block2 = Hybrid_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.w2 = torch.nn.Parameter(torch.ones(1))
        self.BN2 = BatchNorm(64, track_running_stats=False)
        
        self.block3 = Hybrid_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.w3 = torch.nn.Parameter(torch.ones(1))
        self.BN3 = BatchNorm(64, track_running_stats=False)
        
        self.block4 = Hybrid_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.w4 = torch.nn.Parameter(torch.ones(1))
        self.BN4 = BatchNorm(64, track_running_stats=False)

        self.block5 = Hybrid_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.w5 = torch.nn.Parameter(torch.ones(1))
        self.BN5 = BatchNorm(64, track_running_stats=False)

        self.block6 = Hybrid_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.w6 = torch.nn.Parameter(torch.ones(1))
        self.BN6 = BatchNorm(64, track_running_stats=False)

        self.block7 = Hybrid_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.w7 = torch.nn.Parameter(torch.ones(1))
        self.BN7 = BatchNorm(64, track_running_stats=False)

        self.block8 = Hybrid_Block(64, 64, GAT_heads=8, edge_dim=6, MLP_dim=64, drop_prob=.01, GAT_aggr=GAT_aggr, GIN_aggr=GIN_aggr)
        self.w8 = torch.nn.Parameter(torch.ones(1))
        self.BN8 = BatchNorm(64, track_running_stats=False)

        self.post_BN = BatchNorm(576, track_running_stats=False)
        self.postprocess1 = nn.Linear(576, 256)
        self.postprocess2 = nn.Linear(256, 128)
        self.postprocess3 = nn.Linear(128, 64)
        self.postprocess4 = nn.Linear(64, 32)
        self.postprocess5 = nn.Linear(32, 16)
        self.postprocess6 = nn.Linear(16, output_dim)
        
        self.elu = torch.nn.ELU()
        self.softmax = torch.nn.Softmax(dim=0)

	# Regression head for reconstruction
        self.rcon1 = nn.Linear(64, 64)
        self.rcon2 = nn.Linear(64, 64)
        self.rcon3 = nn.Linear(64, input_dim)

    def forward(self, input):
        x = input.x
        if self.training:
            x += (x.std(dim=0)*self.noise_variance)*torch.randn_like(x)

        x = self.BN0(x)

        x = self.pre_BN1(self.preprocess1(x))
        x = self.elu(x)
        x = self.pre_BN2(self.preprocess2(x))
        x = self.elu(x)
        x = self.GRU0(x)
        x = self.elu(self.pre_BN3(x))

        block1_out = self.block1(x, input.edge_index, input.edge_attr)
        block1_out = torch.add(block1_out * self.w1, x)
        block1_out = self.elu(self.BN1(block1_out))

        block2_out = self.block2(block1_out, input.edge_index, input.edge_attr)
        block2_out = torch.add(block2_out * self.w2, block1_out)
        block2_out = self.elu(self.BN2(block2_out))

        block3_out = self.block3(block2_out, input.edge_index, input.edge_attr)
        block3_out = torch.add(block3_out * self.w3, block2_out)
        block3_out = self.elu(self.BN3(block3_out))

        block4_out = self.block4(block3_out, input.edge_index, input.edge_attr)
        block4_out = torch.add(block4_out * self.w4, block3_out)
        block4_out = self.elu(self.BN4(block4_out))

        block5_out = self.block5(block4_out, input.edge_index, input.edge_attr)
        block5_out = torch.add(block5_out * self.w5, block4_out)
        block5_out = self.elu(self.BN5(block5_out))

        block6_out = self.block6(block5_out, input.edge_index, input.edge_attr)
        block6_out = torch.add(block6_out * self.w6, block5_out)
        block6_out = self.elu(self.BN6(block6_out))

        block7_out = self.block7(block6_out, input.edge_index, input.edge_attr)
        block7_out = torch.add(block7_out * self.w7, block6_out)
        block7_out = self.elu(self.BN7(block7_out))

        block8_out = self.block8(block7_out, input.edge_index, input.edge_attr)
        block8_out = torch.add(block8_out * self.w8, block7_out)
        block8_out = self.elu(self.BN8(block8_out))

        combined = torch.cat((block1_out, block2_out, block3_out, block4_out,
         block5_out, block6_out, block7_out, block8_out, x), dim=-1)

        combined = self.post_BN(combined)

        x = self.elu(self.postprocess1(combined))
        x = self.elu(self.postprocess2(x))
        x = self.elu(self.postprocess3(x))
        x = self.elu(self.postprocess4(x))
        x = self.elu(self.postprocess5(x))
        x = self.postprocess6(x)

	# Regression Head for Reconstruction Loss
        rcon_output = self.elu(self.rcon1(block8_out))
        rcon_output = self.elu(self.rcon2(rcon_output))
        rcon_output = self.rcon3(rcon_output)

        return x, rcon_output

# class Two_Track_GIN_GAT_No_Added_Concat(nn.Module):
#     def __init__(self, input_dim, output_dim=2, drop_prob=0.1, GAT_aggr="mean", GIN_aggr="add"):
#         # No need for bias in GAT Convs due to batch norms
#         super(Two_Track_GIN_GAT_No_Added_Concat, self).__init__() 
#         self.preprocess1 = nn.Linear(input_dim, 48, bias=False)
#         self.pre_BN1 = BatchNorm(48)
#         self.preprocess2 = nn.Linear(48, 64, bias=False)
#         self.pre_BN2 = BatchNorm(64)
        
#         # Left Track
#         self.left_GAT_1 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=GAT_aggr)
#         self.left_BN1 = BatchNorm(64)
        
#         self.left_GAT_2 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=GAT_aggr)
#         self.left_BN2 = BatchNorm(64)

#         self.left_GAT_3 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=GAT_aggr)
#         self.left_BN3 = BatchNorm(64)

#         self.left_GAT_4 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=GAT_aggr)
#         self.left_BN4 = BatchNorm(64)
         
#         # Right Track
#         self.right_GAT_1 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
#         self.right_BN1 = BatchNorm(64)
        
#         self.right_GAT_2 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
#         self.right_BN2 = BatchNorm(64)

#         self.right_GAT_3 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
#         self.right_BN3 = BatchNorm(64)

#         self.right_GAT_4 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
#         self.right_BN4 = BatchNorm(64)

#         self.postprocess1 = nn.Linear(512, 256)
#         self.postprocess2 = nn.Linear(256, 128)
#         self.postprocess3 = nn.Linear(128, 64)
#         self.postprocess4 = nn.Linear(64, 32)
#         self.postprocess5 = nn.Linear(32, 16)
#         self.postprocess6 = nn.Linear(16, output_dim)

#         self.elu = torch.nn.ELU()
#         self.softmax = torch.nn.Softmax(dim=0)

#     def forward(self, input):
#         x = self.pre_BN1(self.preprocess1(input.x))
#         x = self.elu(x)
#         x = self.pre_BN2(self.preprocess2(x))
#         x = self.elu(x)
        
#         # Left Track
#         left_block_1_out = self.left_GAT_1(x, input.edge_index)
#         left_block_1_out = self.elu(self.left_BN1(torch.add(left_block_1_out, x)))

#         left_block_2_out = self.left_GAT_2(left_block_1_out,input.edge_index)
#         left_block_2_out = self.elu(self.left_BN2(torch.add(left_block_2_out, left_block_1_out)))              # Resnet style skip connection
        
#         left_block_3_out = self.left_GAT_3(left_block_2_out,input.edge_index)
#         left_block_3_out = self.elu(self.left_BN3(torch.add(left_block_3_out, left_block_2_out)))

#         left_block_4_out = self.left_GAT_4(left_block_3_out,input.edge_index)
#         left_block_4_out = self.elu(self.left_BN4(torch.add(left_block_4_out, left_block_3_out)))
        
#         # Right Track
#         right_block_1_out = self.right_GAT_1(x, input.edge_index)
#         right_block_1_out = self.elu(self.right_BN1(torch.add(right_block_1_out, x)))

#         right_block_2_out = self.right_GAT_2(right_block_1_out,input.edge_index)
#         right_block_2_out = self.elu(self.right_BN2(torch.add(right_block_2_out, right_block_1_out)))              # Resnet style skip connection
        
#         right_block_3_out = self.right_GAT_3(right_block_2_out,input.edge_index)
#         right_block_3_out = self.elu(self.right_BN3(torch.add(right_block_3_out, right_block_2_out)))

#         right_block_4_out = self.right_GAT_4(right_block_3_out,input.edge_index)
#         right_block_4_out = self.elu(self.right_BN4(torch.add(right_block_4_out, right_block_3_out)))
        
#         combined = torch.cat((left_block_4_out,right_block_4_out,left_block_3_out,right_block_3_out,left_block_2_out,right_block_2_out,left_block_1_out,right_block_1_out), dim=-1)
        
#         x = self.elu(self.postprocess1(combined))
#         x = self.elu(self.postprocess2(x))
#         x = self.elu(self.postprocess3(x))
#         x = self.elu(self.postprocess4(x)) 
#         x = self.elu(self.postprocess5(x))
#         x = self.postprocess6(x)

#         return x

# class Two_Track_GIN_GAT(nn.Module):
#     def __init__(self, input_dim, output_dim=2, drop_prob=0.1, GAT_aggr="mean", GIN_aggr="add"):
#         # No need for bias in GAT Convs due to batch norms
#         super(Two_Track_GIN_GAT, self).__init__() 
#         self.preprocess1 = nn.Linear(input_dim, 48, bias=False)
#         self.pre_BN1 = BatchNorm(48)
#         self.preprocess2 = nn.Linear(48, 64, bias=False)
#         self.pre_BN2 = BatchNorm(64)
        
#         # Left Track
#         self.left_GAT_1 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=GAT_aggr)
#         self.left_BN1 = BatchNorm(64)
        
#         self.left_GAT_2 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=GAT_aggr)
#         self.left_BN2 = BatchNorm(64)

#         self.left_GAT_3 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=GAT_aggr)
#         self.left_BN3 = BatchNorm(64)

#         self.left_GAT_4 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=GAT_aggr)
#         self.left_BN4 = BatchNorm(64)
         
#         # Right Track
#         self.right_GAT_1 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
#         self.right_BN1 = BatchNorm(64)
        
#         self.right_GAT_2 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
#         self.right_BN2 = BatchNorm(64)

#         self.right_GAT_3 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
#         self.right_BN3 = BatchNorm(64)

#         self.right_GAT_4 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
#         self.right_BN4 = BatchNorm(64)

#         self.post_BN = BatchNorm(576)
#         self.postprocess1 = nn.Linear(576, 256)
#         self.postprocess2 = nn.Linear(256, 128)
#         self.postprocess3 = nn.Linear(128, 64)
#         self.postprocess4 = nn.Linear(64, 32)
#         self.postprocess5 = nn.Linear(32, 16)
#         self.postprocess6 = nn.Linear(16, output_dim)

#         self.elu = torch.nn.ELU()
#         self.softmax = torch.nn.Softmax(dim=0)

#     def forward(self, input):
#         x = self.pre_BN1(self.preprocess1(input.x))
#         x = self.elu(x)
#         x = self.pre_BN2(self.preprocess2(x))
#         x = self.elu(x)
        
#         # Left Track
#         left_block_1_out = self.left_GAT_1(x, input.edge_index)
#         left_block_1_out = self.elu(self.left_BN1(torch.add(left_block_1_out, x)))

#         left_block_2_out = self.left_GAT_2(left_block_1_out,input.edge_index)
#         left_block_2_out = self.elu(self.left_BN2(torch.add(left_block_2_out, left_block_1_out)))              # Resnet style skip connection
        
#         left_block_3_out = self.left_GAT_3(left_block_2_out,input.edge_index)
#         left_block_3_out = self.elu(self.left_BN3(torch.add(left_block_3_out, left_block_2_out)))

#         left_block_4_out = self.left_GAT_4(left_block_3_out,input.edge_index)
#         left_block_4_out = self.elu(self.left_BN4(torch.add(left_block_4_out, left_block_3_out)))
        
#         # Right Track
#         right_block_1_out = self.right_GAT_1(x, input.edge_index)
#         right_block_1_out = self.elu(self.right_BN1(torch.add(right_block_1_out, x)))

#         right_block_2_out = self.right_GAT_2(right_block_1_out,input.edge_index)
#         right_block_2_out = self.elu(self.right_BN2(torch.add(right_block_2_out, right_block_1_out)))              # Resnet style skip connection
        
#         right_block_3_out = self.right_GAT_3(right_block_2_out,input.edge_index)
#         right_block_3_out = self.elu(self.right_BN3(torch.add(right_block_3_out, right_block_2_out)))

#         right_block_4_out = self.right_GAT_4(right_block_3_out,input.edge_index)
#         right_block_4_out = self.elu(self.right_BN4(torch.add(right_block_4_out, right_block_3_out)))
        
#         combined = torch.cat((left_block_4_out,right_block_4_out,left_block_3_out,right_block_3_out,left_block_2_out,right_block_2_out,left_block_1_out,right_block_1_out,x), dim=-1)

#         combined = self.post_BN(combined)
        
#         x = self.elu(self.postprocess1(combined))
#         x = self.elu(self.postprocess2(x))
#         x = self.elu(self.postprocess3(x))
#         x = self.elu(self.postprocess4(x))
#         x = self.elu(self.postprocess5(x))
#         x = self.postprocess6(x)

#         return x

# class Two_Track_GIN_GAT_Extra_BN(nn.Module):
#     def __init__(self, input_dim, output_dim=2, drop_prob=0.1, GAT_aggr="mean", GIN_aggr="add"):
#         # No need for bias in GAT Convs due to batch norms
#         super(Two_Track_GIN_GAT_Extra_BN, self).__init__() 
#         self.preprocess1 = nn.Linear(input_dim, 48, bias=False)
#         self.pre_BN1 = BatchNorm(48)
#         self.preprocess2 = nn.Linear(48, 64, bias=False)
#         self.pre_BN2 = BatchNorm(64)
        
#         # Left Track
#         self.left_GAT_1 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=GAT_aggr)
#         self.left_BN1 = BatchNorm(64)
        
#         self.left_GAT_2 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=GAT_aggr)
#         self.left_BN2 = BatchNorm(64)

#         self.left_GAT_3 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=GAT_aggr)
#         self.left_BN3 = BatchNorm(64)

#         self.left_GAT_4 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=GAT_aggr)
#         self.left_BN4 = BatchNorm(64)
         
#         # Right Track
#         self.right_GAT_1 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
#         self.right_BN1 = BatchNorm(64)
        
#         self.right_GAT_2 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
#         self.right_BN2 = BatchNorm(64)

#         self.right_GAT_3 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
#         self.right_BN3 = BatchNorm(64)

#         self.right_GAT_4 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
#         self.right_BN4 = BatchNorm(64)

#         self.post_BN = BatchNorm(512)
#         self.postprocess1 = nn.Linear(512, 256)
#         self.postprocess2 = nn.Linear(256, 128)
#         self.postprocess3 = nn.Linear(128, 64)
#         self.postprocess4 = nn.Linear(64, 32)
#         self.postprocess5 = nn.Linear(32, 16)
#         self.postprocess6 = nn.Linear(16, output_dim)

#         self.elu = torch.nn.ELU()
#         self.softmax = torch.nn.Softmax(dim=0)

#     def forward(self, input):
#         x = self.pre_BN1(self.preprocess1(input.x))
#         x = self.elu(x)
#         x = self.pre_BN2(self.preprocess2(x))
#         x = self.elu(x)
        
#         # Left Track
#         left_block_1_out = self.left_GAT_1(x, input.edge_index)
#         left_block_1_out = self.elu(self.left_BN1(torch.add(left_block_1_out, x)))

#         left_block_2_out = self.left_GAT_2(left_block_1_out,input.edge_index)
#         left_block_2_out = self.elu(self.left_BN2(torch.add(left_block_2_out, left_block_1_out)))              # Resnet style skip connection
        
#         left_block_3_out = self.left_GAT_3(left_block_2_out,input.edge_index)
#         left_block_3_out = self.elu(self.left_BN3(torch.add(left_block_3_out, left_block_2_out)))

#         left_block_4_out = self.left_GAT_4(left_block_3_out,input.edge_index)
#         left_block_4_out = self.elu(self.left_BN4(torch.add(left_block_4_out, left_block_3_out)))
        
#         # Right Track
#         right_block_1_out = self.right_GAT_1(x, input.edge_index)
#         right_block_1_out = self.elu(self.right_BN1(torch.add(right_block_1_out, x)))

#         right_block_2_out = self.right_GAT_2(right_block_1_out,input.edge_index)
#         right_block_2_out = self.elu(self.right_BN2(torch.add(right_block_2_out, right_block_1_out)))              # Resnet style skip connection
        
#         right_block_3_out = self.right_GAT_3(right_block_2_out,input.edge_index)
#         right_block_3_out = self.elu(self.right_BN3(torch.add(right_block_3_out, right_block_2_out)))

#         right_block_4_out = self.right_GAT_4(right_block_3_out,input.edge_index)
#         right_block_4_out = self.elu(self.right_BN4(torch.add(right_block_4_out, right_block_3_out)))
        
#         combined = self.post_BN(torch.cat((left_block_4_out,right_block_4_out,left_block_3_out,right_block_3_out,left_block_2_out,right_block_2_out,left_block_1_out,right_block_1_out), dim=-1))
        
#         x = self.elu(self.postprocess1(combined))
#         x = self.elu(self.postprocess2(x))
#         x = self.elu(self.postprocess3(x))
#         x = self.elu(self.postprocess4(x))
#         x = self.elu(self.postprocess5(x))
#         x = self.postprocess6(x)

#        return x  

# class Two_Track_GAT_GAT(nn.Module):
#     def __init__(self, input_dim, output_dim=2, drop_prob=0.1, left_aggr="mean", right_aggr="add"):
#         # No need for bias in GAT Convs due to batch norms
#         super(Two_Track_GAT_GAT, self).__init__() 
#         self.preprocess1 = nn.Linear(input_dim, 48, bias=False)
#         self.pre_BN1 = BatchNorm(48)
#         self.preprocess2 = nn.Linear(48, 64, bias=False)
#         self.pre_BN2 = BatchNorm(64)
        
#         # Left Track
#         self.left_GAT_1 = GINConv(MLP(3, 64, 64, 64), aggr=left_aggr)
#         self.left_BN1 = BatchNorm(64)
        
#         self.left_GAT_2 = GINConv(MLP(3, 64, 64, 64), aggr=left_aggr)
#         self.left_BN2 = BatchNorm(64)

#         self.left_GAT_3 = GINConv(MLP(3, 64, 64, 64), aggr=left_aggr)
#         self.left_BN3 = BatchNorm(64)

#         self.left_GAT_4 = GINConv(MLP(3, 64, 64, 64), aggr=left_aggr)
#         self.left_BN4 = BatchNorm(64)
         
#         # Right Track
#         self.right_GAT_1 = GINConv(MLP(3, 64, 64, 64), aggr=right_aggr)
#         self.right_BN1 = BatchNorm(64)
        
#         self.right_GAT_2 = GINConv(MLP(3, 64, 64, 64), aggr=right_aggr)
#         self.right_BN2 = BatchNorm(64)

#         self.right_GAT_3 = GINConv(MLP(3, 64, 64, 64), aggr=right_aggr)
#         self.right_BN3 = BatchNorm(64)

#         self.right_GAT_4 = GINConv(MLP(3, 64, 64, 64), aggr=right_aggr)
#         self.right_BN4 = BatchNorm(64)

#         self.postprocess1 = nn.Linear(512, 256)
#         self.postprocess2 = nn.Linear(256, 128)
#         self.postprocess3 = nn.Linear(128, 64)
#         self.postprocess4 = nn.Linear(64, 32)
#         self.postprocess5 = nn.Linear(32, 16)
#         self.postprocess6 = nn.Linear(16, output_dim)

#         self.elu = torch.nn.ELU()
#         self.softmax = torch.nn.Softmax(dim=0)

#     def forward(self, input):
#         x = self.pre_BN1(self.preprocess1(input.x))
#         x = self.elu(x)
#         x = self.pre_BN2(self.preprocess2(x))
#         x = self.elu(x)
        
#         # Left Track
#         left_block_1_out = self.left_GAT_1(x, input.edge_index)
#         left_block_1_out = self.elu(self.left_BN1(torch.add(left_block_1_out, x)))

#         left_block_2_out = self.left_GAT_2(left_block_1_out,input.edge_index)
#         left_block_2_out = self.elu(self.left_BN2(torch.add(left_block_2_out, left_block_1_out)))              # Resnet style skip connection
        
#         left_block_3_out = self.left_GAT_3(left_block_2_out,input.edge_index)
#         left_block_3_out = self.elu(self.left_BN3(torch.add(left_block_3_out, left_block_2_out)))

#         left_block_4_out = self.left_GAT_4(left_block_3_out,input.edge_index)
#         left_block_4_out = self.elu(self.left_BN4(torch.add(left_block_4_out, left_block_3_out)))
        
#         # Right Track
#         right_block_1_out = self.right_GAT_1(x, input.edge_index)
#         right_block_1_out = self.elu(self.right_BN1(torch.add(right_block_1_out, x)))

#         right_block_2_out = self.right_GAT_2(right_block_1_out,input.edge_index)
#         right_block_2_out = self.elu(self.right_BN2(torch.add(right_block_2_out, right_block_1_out)))              # Resnet style skip connection
        
#         right_block_3_out = self.right_GAT_3(right_block_2_out,input.edge_index)
#         right_block_3_out = self.elu(self.right_BN3(torch.add(right_block_3_out, right_block_2_out)))

#         right_block_4_out = self.right_GAT_4(right_block_3_out,input.edge_index)
#         right_block_4_out = self.elu(self.right_BN4(torch.add(right_block_4_out, right_block_3_out)))
        
#         combined = torch.cat((left_block_4_out,right_block_4_out,left_block_3_out,right_block_3_out,left_block_2_out,right_block_2_out,left_block_1_out,right_block_1_out), dim=-1)
        
#         x = self.elu(self.postprocess1(combined))
#         x = self.elu(self.postprocess2(x))
#         x = self.elu(self.postprocess3(x))
#         x = self.elu(self.postprocess4(x))
#         x = self.elu(self.postprocess5(x))
#         x = self.postprocess6(x)

#         return x

# class GATModelv1(nn.Module):
#     def __init__(self, input_dim, output_dim=2):
#         # No need for bias in GAT Convs due to batch norms
#         super(GATModelv1, self).__init__()
#         self.preprocess1 = nn.Linear(input_dim, 128, bias=False)
#         self.BN0 = BatchNorm(128)
        
#         self.GAT_1 = GATv2Conv(128, 8, heads=8, bias=False)
#         self.BN1 = BatchNorm(64)
        
#         self.preprocess2 = nn.Linear(64, 48)
#         self.GAT_2 = GATv2Conv(48, 6, heads=8, bias=False)
#         self.BN2 = BatchNorm(48)

#         self.preprocess3 = nn.Linear(48, 32)
#         self.GAT_3 = GATv2Conv(32, 5, heads=5, bias=False)
#         self.BN3 = BatchNorm(25)

#         self.postprocess1 = nn.Linear(25, output_dim)

#         self.elu = torch.nn.ELU()
#         self.softmax = torch.nn.Softmax(dim=0)
        
#     def forward(self, input):
#         x = self.BN0(self.elu(self.preprocess1(input.x)))
        
#         x = self.BN1(self.GAT_1(x, input.edge_index))
        
#         x = self.elu(self.preprocess2(x))
#         x = self.BN2(self.GAT_2(x,input.edge_index))

#         x = self.elu(self.preprocess3(x))
#         x = self.BN3(self.GAT_3(x,input.edge_index))
        
#         x = self.postprocess1(x)
#         return x


# class GATModelv2(nn.Module):
#     def __init__(self, input_dim, output_dim=2):
#         # No need for bias in GAT Convs due to batch norms
#         super(GATModelv2, self).__init__()
#         self.preprocess1 = nn.Linear(input_dim, 64, bias=False)
#         self.BN0 = BatchNorm(64)
        
#         self.GAT_1 = GATv2Conv(64, 8, heads=8, bias=False)
#         self.BN1 = BatchNorm(64)
        
#         self.GAT_2 = GATv2Conv(64, 8, heads=8, bias=False)
#         self.BN2 = BatchNorm(64)

#         self.GAT_3 = GATv2Conv(64, 8, heads=8, bias=False)
#         self.BN3 = BatchNorm(64)

#         self.GAT_4 = GATv2Conv(64, 8, heads=8, bias=False)
#         self.BN4 = BatchNorm(64)

#         self.postprocess1 = nn.Linear(64, output_dim)

#         self.elu = torch.nn.ELU()
#         self.softmax = torch.nn.Softmax(dim=0)
         
#     def forward(self, input):
#         x = self.BN0(self.preprocess1(input.x))
#         x = self.elu(x)
        
#         block_1_out = self.BN1(self.GAT_1(x, input.edge_index))
#         block_1_out = self.elu(torch.add(block_1_out, x))

#         block_2_out = self.BN2(self.GAT_2(block_1_out,input.edge_index))
#         block_2_out = self.elu(torch.add(block_2_out, block_1_out))              # DenseNet style skip connection
        
#         block_3_out = self.BN3(self.GAT_3(block_2_out,input.edge_index))
#         block_3_out = self.elu(torch.add(block_3_out, block_2_out))

#         block_4_out = self.BN4(self.GAT_4(block_3_out,input.edge_index))
#         block_4_out = self.elu(torch.add(block_4_out, block_3_out))

        
#         x = self.postprocess1(block_4_out)
#         return x

# class One_Track_GATModel(nn.Module):
#     def __init__(self, input_dim, output_dim=2, drop_prob=0.1):
#         # No need for bias in GAT Convs due to batch norms
#         super(One_Track_GATModel, self).__init__()
#         self.preprocess1 = nn.Linear(input_dim, 64, bias=False)
#         self.BN0 = BatchNorm(64)
        
#         self.GAT_1 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob)
#         self.BN1 = BatchNorm(64)
        
#         self.GAT_2 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob)
#         self.BN2 = BatchNorm(64)

#         self.GAT_3 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob)
#         self.BN3 = BatchNorm(64)

#         self.GAT_4 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob)
#         self.BN4 = BatchNorm(64)

#         self.postprocess1 = nn.Linear(64, 48)
#         self.postprocess2 = nn.Linear(48, 32)
#         self.postprocess3 = nn.Linear(32, 16)
#         self.postprocess4 = nn.Linear(16, output_dim)

#         self.elu = torch.nn.ELU()
#         self.softmax = torch.nn.Softmax(dim=0)
         
#     def forward(self, input):
#         x = self.BN0(self.preprocess1(input.x))
#         x = self.elu(x)
        
#         block_1_out = self.GAT_1(x, input.edge_index)
#         block_1_out = self.elu(self.BN1(torch.add(block_1_out, x, aggr="add")))

#         block_2_out = self.GAT_2(block_1_out,input.edge_index)
#         block_2_out = self.elu(self.BN2(torch.add(block_2_out, block_1_out, aggr="add")))              # Resnet style skip connection
        
#         block_3_out = self.GAT_3(block_2_out,input.edge_index)
#         block_3_out = self.elu(self.BN3(torch.add(block_3_out, block_2_out, aggr="add")))

#         block_4_out = self.GAT_4(block_3_out,input.edge_index)
#         block_4_out = self.elu(self.BN4(torch.add(block_4_out, block_3_out, aggr="add")))

        
#         x = self.elu(self.postprocess1(block_4_out))
#         x = self.elu(self.postprocess2(x))
#         x = self.elu(self.postprocess3(x))
#         x = self.postprocess4(x)

#         return x

# class Two_Track_GATModel(nn.Module):
#     def __init__(self, input_dim, output_dim=2, drop_prob=0.1, left_aggr="add", right_aggr="mean"):
#         # No need for bias in GAT Convs due to batch norms
#         super(Two_Track_GATModel, self).__init__()
#         self.preprocess1 = nn.Linear(input_dim, 48, bias=False)
#         self.pre_BN1 = BatchNorm(48)
#         self.preprocess2 = nn.Linear(48, 64, bias=False)
#         self.pre_BN2 = BatchNorm(64)
        
#         # Left Track
#         self.left_GAT_1 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=left_aggr)
#         self.left_BN1 = BatchNorm(64)
        
#         self.left_GAT_2 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=left_aggr)
#         self.left_BN2 = BatchNorm(64)

#         self.left_GAT_3 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=left_aggr)
#         self.left_BN3 = BatchNorm(64)

#         self.left_GAT_4 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=left_aggr)
#         self.left_BN4 = BatchNorm(64)
         
#         # Right Track
#         self.right_GAT_1 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=right_aggr)
#         self.right_BN1 = BatchNorm(64)
        
#         self.right_GAT_2 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=right_aggr)
#         self.right_BN2 = BatchNorm(64)

#         self.right_GAT_3 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=right_aggr)
#         self.right_BN3 = BatchNorm(64)

#         self.right_GAT_4 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=right_aggr)
#         self.right_BN4 = BatchNorm(64)

#         self.postprocess1 = nn.Linear(128, 64)
#         self.postprocess2 = nn.Linear(64, 32)
#         self.postprocess3 = nn.Linear(32, 16)
#         self.postprocess4 = nn.Linear(16, output_dim)

#         self.elu = torch.nn.ELU()
#         self.softmax = torch.nn.Softmax(dim=0)

#     def forward(self, input):
#         x = self.pre_BN1(self.preprocess1(input.x))
#         x = self.elu(x)
#         x = self.pre_BN2(self.preprocess2(x))
#         x = self.elu(x)
        
#         # Left Track
#         left_block_1_out = self.left_GAT_1(x, input.edge_index)
#         left_block_1_out = self.elu(self.left_BN1(torch.add(left_block_1_out, x)))

#         left_block_2_out = self.left_GAT_2(left_block_1_out,input.edge_index)
#         left_block_2_out = self.elu(self.left_BN2(torch.add(left_block_2_out, left_block_1_out)))              # Resnet style skip connection
        
#         left_block_3_out = self.left_GAT_3(left_block_2_out,input.edge_index)
#         left_block_3_out = self.elu(self.left_BN3(torch.add(left_block_3_out, left_block_2_out)))

#         left_block_4_out = self.left_GAT_4(left_block_3_out,input.edge_index)
#         left_block_4_out = self.elu(self.left_BN4(torch.add(left_block_4_out, left_block_3_out)))
        
#         # Right Track
#         right_block_1_out = self.right_GAT_1(x, input.edge_index)
#         right_block_1_out = self.elu(self.right_BN1(torch.add(right_block_1_out, x)))

#         right_block_2_out = self.right_GAT_2(right_block_1_out,input.edge_index)
#         right_block_2_out = self.elu(self.right_BN2(torch.add(right_block_2_out, right_block_1_out)))              # Resnet style skip connection
        
#         right_block_3_out = self.right_GAT_3(right_block_2_out,input.edge_index)
#         right_block_3_out = self.elu(self.right_BN3(torch.add(right_block_3_out, right_block_2_out)))

#         right_block_4_out = self.right_GAT_4(right_block_3_out,input.edge_index)
#         right_block_4_out = self.elu(self.right_BN4(torch.add(right_block_4_out, right_block_3_out)))
        
#         combined = torch.cat((left_block_4_out,right_block_4_out), dim=-1)
        
#         x = self.elu(self.postprocess1(combined))
#         x = self.elu(self.postprocess2(x))
#         x = self.elu(self.postprocess3(x))
#         x = self.postprocess4(x)

#         return x

# class Two_Track_JK_GATModel(nn.Module):
#     def __init__(self, input_dim, output_dim=2, drop_prob=0.1, left_aggr="add", right_aggr="mean"):
#         # No need for bias in GAT Convs due to batch norms
#         super(Two_Track_JK_GATModel, self).__init__() 
#         self.preprocess1 = nn.Linear(input_dim, 48, bias=False)
#         self.pre_BN1 = BatchNorm(48)
#         self.preprocess2 = nn.Linear(48, 64, bias=False)
#         self.pre_BN2 = BatchNorm(64)
        
#         # Left Track
#         self.left_GAT_1 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=left_aggr)
#         self.left_BN1 = BatchNorm(64)
        
#         self.left_GAT_2 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=left_aggr)
#         self.left_BN2 = BatchNorm(64)

#         self.left_GAT_3 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=left_aggr)
#         self.left_BN3 = BatchNorm(64)

#         self.left_GAT_4 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=left_aggr)
#         self.left_BN4 = BatchNorm(64)
         
#         # Right Track
#         self.right_GAT_1 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=right_aggr)
#         self.right_BN1 = BatchNorm(64)
        
#         self.right_GAT_2 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=right_aggr)
#         self.right_BN2 = BatchNorm(64)

#         self.right_GAT_3 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=right_aggr)
#         self.right_BN3 = BatchNorm(64)

#         self.right_GAT_4 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=right_aggr)
#         self.right_BN4 = BatchNorm(64)

#         self.postprocess1 = nn.Linear(512, 256)
#         self.postprocess2 = nn.Linear(256, 128)
#         self.postprocess3 = nn.Linear(128, 64)
#         self.postprocess4 = nn.Linear(64, 32)
#         self.postprocess5 = nn.Linear(32, 16)
#         self.postprocess6 = nn.Linear(16, output_dim)

#         self.elu = torch.nn.ELU()
#         self.softmax = torch.nn.Softmax(dim=0)

#     def forward(self, input):
#         x = self.pre_BN1(self.preprocess1(input.x))
#         x = self.elu(x)
#         x = self.pre_BN2(self.preprocess2(x))
#         x = self.elu(x)
        
#         # Left Track
#         left_block_1_out = self.left_GAT_1(x, input.edge_index)
#         left_block_1_out = self.elu(self.left_BN1(torch.add(left_block_1_out, x)))

#         left_block_2_out = self.left_GAT_2(left_block_1_out,input.edge_index)
#         left_block_2_out = self.elu(self.left_BN2(torch.add(left_block_2_out, left_block_1_out)))              # Resnet style skip connection
        
#         left_block_3_out = self.left_GAT_3(left_block_2_out,input.edge_index)
#         left_block_3_out = self.elu(self.left_BN3(torch.add(left_block_3_out, left_block_2_out)))

#         left_block_4_out = self.left_GAT_4(left_block_3_out,input.edge_index)
#         left_block_4_out = self.elu(self.left_BN4(torch.add(left_block_4_out, left_block_3_out)))
        
#         # Right Track
#         right_block_1_out = self.right_GAT_1(x, input.edge_index)
#         right_block_1_out = self.elu(self.right_BN1(torch.add(right_block_1_out, x)))

#         right_block_2_out = self.right_GAT_2(right_block_1_out,input.edge_index)
#         right_block_2_out = self.elu(self.right_BN2(torch.add(right_block_2_out, right_block_1_out)))              # Resnet style skip connection
        
#         right_block_3_out = self.right_GAT_3(right_block_2_out,input.edge_index)
#         right_block_3_out = self.elu(self.right_BN3(torch.add(right_block_3_out, right_block_2_out)))

#         right_block_4_out = self.right_GAT_4(right_block_3_out,input.edge_index)
#         right_block_4_out = self.elu(self.right_BN4(torch.add(right_block_4_out, right_block_3_out)))
        
#         combined = torch.cat((left_block_4_out,right_block_4_out,left_block_3_out,right_block_3_out,left_block_2_out,right_block_2_out,left_block_1_out,right_block_1_out), dim=-1)
        
#         x = self.elu(self.postprocess1(combined))
#         x = self.elu(self.postprocess2(x))
#         x = self.elu(self.postprocess3(x))
#         x = self.elu(self.postprocess4(x))
#         x = self.elu(self.postprocess5(x))
#         x = self.postprocess6(x)

#         return x
    
