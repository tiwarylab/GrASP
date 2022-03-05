import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, GINConv
from torch_geometric.nn.norm import BatchNorm
import torch.nn.functional as F


class GATModelv1(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        # No need for bias in GAT Convs due to batch norms
        super(GATModelv1, self).__init__()
        self.preprocess1 = nn.Linear(input_dim, 128, bias=False)
        self.BN0 = BatchNorm(128)
        
        self.GAT_1 = GATv2Conv(128, 8, heads=8, bias=False)
        self.BN1 = BatchNorm(64)
        
        self.preprocess2 = nn.Linear(64, 48)
        self.GAT_2 = GATv2Conv(48, 6, heads=8, bias=False)
        self.BN2 = BatchNorm(48)

        self.preprocess3 = nn.Linear(48, 32)
        self.GAT_3 = GATv2Conv(32, 5, heads=5, bias=False)
        self.BN3 = BatchNorm(25)

        self.postprocess1 = nn.Linear(25, output_dim)

        self.elu = torch.nn.ELU()
        self.softmax = torch.nn.Softmax(dim=0)
        
    def forward(self, input):
        x = self.BN0(self.elu(self.preprocess1(input.x)))
        
        x = self.BN1(self.GAT_1(x, input.edge_index))
        
        x = self.elu(self.preprocess2(x))
        x = self.BN2(self.GAT_2(x,input.edge_index))

        x = self.elu(self.preprocess3(x))
        x = self.BN3(self.GAT_3(x,input.edge_index))
        
        x = self.postprocess1(x)
        return x


class GATModelv2(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        # No need for bias in GAT Convs due to batch norms
        super(GATModelv2, self).__init__()
        self.preprocess1 = nn.Linear(input_dim, 64, bias=False)
        self.BN0 = BatchNorm(64)
        
        self.GAT_1 = GATv2Conv(64, 8, heads=8, bias=False)
        self.BN1 = BatchNorm(64)
        
        self.GAT_2 = GATv2Conv(64, 8, heads=8, bias=False)
        self.BN2 = BatchNorm(64)

        self.GAT_3 = GATv2Conv(64, 8, heads=8, bias=False)
        self.BN3 = BatchNorm(64)

        self.GAT_4 = GATv2Conv(64, 8, heads=8, bias=False)
        self.BN4 = BatchNorm(64)

        self.postprocess1 = nn.Linear(64, output_dim)

        self.elu = torch.nn.ELU()
        self.softmax = torch.nn.Softmax(dim=0)
         
    def forward(self, input):
        x = self.BN0(self.preprocess1(input.x))
        x = self.elu(x)
        
        block_1_out = self.BN1(self.GAT_1(x, input.edge_index))
        block_1_out = self.elu(torch.add(block_1_out, x))

        block_2_out = self.BN2(self.GAT_2(block_1_out,input.edge_index))
        block_2_out = self.elu(torch.add(block_2_out, block_1_out))              # DenseNet style skip connection
        
        block_3_out = self.BN3(self.GAT_3(block_2_out,input.edge_index))
        block_3_out = self.elu(torch.add(block_3_out, block_2_out))

        block_4_out = self.BN4(self.GAT_4(block_3_out,input.edge_index))
        block_4_out = self.elu(torch.add(block_4_out, block_3_out))

        
        x = self.postprocess1(block_4_out)
        return x

class One_Track_GATModel(nn.Module):
    def __init__(self, input_dim, output_dim=2, drop_prob=0.1):
        # No need for bias in GAT Convs due to batch norms
        super(One_Track_GATModel, self).__init__()
        self.preprocess1 = nn.Linear(input_dim, 64, bias=False)
        self.BN0 = BatchNorm(64)
        
        self.GAT_1 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob)
        self.BN1 = BatchNorm(64)
        
        self.GAT_2 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob)
        self.BN2 = BatchNorm(64)

        self.GAT_3 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob)
        self.BN3 = BatchNorm(64)

        self.GAT_4 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob)
        self.BN4 = BatchNorm(64)

        self.postprocess1 = nn.Linear(64, 48)
        self.postprocess2 = nn.Linear(48, 32)
        self.postprocess3 = nn.Linear(32, 16)
        self.postprocess4 = nn.Linear(16, output_dim)

        self.elu = torch.nn.ELU()
        self.softmax = torch.nn.Softmax(dim=0)
         
    def forward(self, input):
        x = self.BN0(self.preprocess1(input.x))
        x = self.elu(x)
        
        block_1_out = self.GAT_1(x, input.edge_index)
        block_1_out = self.elu(self.BN1(torch.add(block_1_out, x, aggr="add")))

        block_2_out = self.GAT_2(block_1_out,input.edge_index)
        block_2_out = self.elu(self.BN2(torch.add(block_2_out, block_1_out, aggr="add")))              # Resnet style skip connection
        
        block_3_out = self.GAT_3(block_2_out,input.edge_index)
        block_3_out = self.elu(self.BN3(torch.add(block_3_out, block_2_out, aggr="add")))

        block_4_out = self.GAT_4(block_3_out,input.edge_index)
        block_4_out = self.elu(self.BN4(torch.add(block_4_out, block_3_out, aggr="add")))

        
        x = self.elu(self.postprocess1(block_4_out))
        x = self.elu(self.postprocess2(x))
        x = self.elu(self.postprocess3(x))
        x = self.postprocess4(x)

        return x

class Two_Track_GATModel(nn.Module):
    def __init__(self, input_dim, output_dim=2, drop_prob=0.1, left_aggr="add", right_aggr="mean"):
        # No need for bias in GAT Convs due to batch norms
        super(Two_Track_GATModel, self).__init__()
        self.preprocess1 = nn.Linear(input_dim, 48, bias=False)
        self.pre_BN1 = BatchNorm(48)
        self.preprocess2 = nn.Linear(48, 64, bias=False)
        self.pre_BN2 = BatchNorm(64)
        
        # Left Track
        self.left_GAT_1 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=left_aggr)
        self.left_BN1 = BatchNorm(64)
        
        self.left_GAT_2 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=left_aggr)
        self.left_BN2 = BatchNorm(64)

        self.left_GAT_3 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=left_aggr)
        self.left_BN3 = BatchNorm(64)

        self.left_GAT_4 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=left_aggr)
        self.left_BN4 = BatchNorm(64)
         
        # Right Track
        self.right_GAT_1 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=right_aggr)
        self.right_BN1 = BatchNorm(64)
        
        self.right_GAT_2 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=right_aggr)
        self.right_BN2 = BatchNorm(64)

        self.right_GAT_3 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=right_aggr)
        self.right_BN3 = BatchNorm(64)

        self.right_GAT_4 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=right_aggr)
        self.right_BN4 = BatchNorm(64)

        self.postprocess1 = nn.Linear(128, 64)
        self.postprocess2 = nn.Linear(64, 32)
        self.postprocess3 = nn.Linear(32, 16)
        self.postprocess4 = nn.Linear(16, output_dim)

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
        
        combined = torch.cat((left_block_4_out,right_block_4_out), dim=-1)
        
        x = self.elu(self.postprocess1(combined))
        x = self.elu(self.postprocess2(x))
        x = self.elu(self.postprocess3(x))
        x = self.postprocess4(x)

        return x

class Two_Track_JK_GATModel(nn.Module):
    def __init__(self, input_dim, output_dim=2, drop_prob=0.1, left_aggr="add", right_aggr="mean"):
        # No need for bias in GAT Convs due to batch norms
        super(Two_Track_JK_GATModel, self).__init__() 
        self.preprocess1 = nn.Linear(input_dim, 48, bias=False)
        self.pre_BN1 = BatchNorm(48)
        self.preprocess2 = nn.Linear(48, 64, bias=False)
        self.pre_BN2 = BatchNorm(64)
        
        # Left Track
        self.left_GAT_1 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=left_aggr)
        self.left_BN1 = BatchNorm(64)
        
        self.left_GAT_2 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=left_aggr)
        self.left_BN2 = BatchNorm(64)

        self.left_GAT_3 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=left_aggr)
        self.left_BN3 = BatchNorm(64)

        self.left_GAT_4 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=left_aggr)
        self.left_BN4 = BatchNorm(64)
         
        # Right Track
        self.right_GAT_1 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=right_aggr)
        self.right_BN1 = BatchNorm(64)
        
        self.right_GAT_2 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=right_aggr)
        self.right_BN2 = BatchNorm(64)

        self.right_GAT_3 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=right_aggr)
        self.right_BN3 = BatchNorm(64)

        self.right_GAT_4 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=right_aggr)
        self.right_BN4 = BatchNorm(64)

        self.postprocess1 = nn.Linear(512, 256)
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
        
        combined = torch.cat((left_block_4_out,right_block_4_out,left_block_3_out,right_block_3_out,left_block_2_out,right_block_2_out,left_block_1_out,right_block_1_out), dim=-1)
        
        x = self.elu(self.postprocess1(combined))
        x = self.elu(self.postprocess2(x))
        x = self.elu(self.postprocess3(x))
        x = self.elu(self.postprocess4(x))
        x = self.elu(self.postprocess5(x))
        x = self.postprocess6(x)

        return x
    
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

class Two_Track_GIN_GAT_No_Added_Concat(nn.Module):
    def __init__(self, input_dim, output_dim=2, drop_prob=0.1, GAT_aggr="mean", GIN_aggr="add"):
        # No need for bias in GAT Convs due to batch norms
        super(Two_Track_GIN_GAT_No_Added_Concat, self).__init__() 
        self.preprocess1 = nn.Linear(input_dim, 48, bias=False)
        self.pre_BN1 = BatchNorm(48)
        self.preprocess2 = nn.Linear(48, 64, bias=False)
        self.pre_BN2 = BatchNorm(64)
        
        # Left Track
        self.left_GAT_1 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=GAT_aggr)
        self.left_BN1 = BatchNorm(64)
        
        self.left_GAT_2 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=GAT_aggr)
        self.left_BN2 = BatchNorm(64)

        self.left_GAT_3 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=GAT_aggr)
        self.left_BN3 = BatchNorm(64)

        self.left_GAT_4 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=GAT_aggr)
        self.left_BN4 = BatchNorm(64)
         
        # Right Track
        self.right_GAT_1 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
        self.right_BN1 = BatchNorm(64)
        
        self.right_GAT_2 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
        self.right_BN2 = BatchNorm(64)

        self.right_GAT_3 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
        self.right_BN3 = BatchNorm(64)

        self.right_GAT_4 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
        self.right_BN4 = BatchNorm(64)

        self.postprocess1 = nn.Linear(512, 256)
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
        
        combined = torch.cat((left_block_4_out,right_block_4_out,left_block_3_out,right_block_3_out,left_block_2_out,right_block_2_out,left_block_1_out,right_block_1_out), dim=-1)
        
        x = self.elu(self.postprocess1(combined))
        x = self.elu(self.postprocess2(x))
        x = self.elu(self.postprocess3(x))
        x = self.elu(self.postprocess4(x))
        x = self.elu(self.postprocess5(x))
        x = self.postprocess6(x)

        return x
    
class Two_Track_GIN_GAT(nn.Module):
    def __init__(self, input_dim, output_dim=2, drop_prob=0.1, GAT_aggr="mean", GIN_aggr="add"):
        # No need for bias in GAT Convs due to batch norms
        super(Two_Track_GIN_GAT, self).__init__() 
        self.preprocess1 = nn.Linear(input_dim, 48, bias=False)
        self.pre_BN1 = BatchNorm(48)
        self.preprocess2 = nn.Linear(48, 64, bias=False)
        self.pre_BN2 = BatchNorm(64)
        
        # Left Track
        self.left_GAT_1 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=GAT_aggr)
        self.left_BN1 = BatchNorm(64)
        
        self.left_GAT_2 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=GAT_aggr)
        self.left_BN2 = BatchNorm(64)

        self.left_GAT_3 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=GAT_aggr)
        self.left_BN3 = BatchNorm(64)

        self.left_GAT_4 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=GAT_aggr)
        self.left_BN4 = BatchNorm(64)
         
        # Right Track
        self.right_GAT_1 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
        self.right_BN1 = BatchNorm(64)
        
        self.right_GAT_2 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
        self.right_BN2 = BatchNorm(64)

        self.right_GAT_3 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
        self.right_BN3 = BatchNorm(64)

        self.right_GAT_4 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
        self.right_BN4 = BatchNorm(64)

        self.post_BN = BatchNorm(576)
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

class Two_Track_GIN_GAT_Extra_BN(nn.Module):
    def __init__(self, input_dim, output_dim=2, drop_prob=0.1, GAT_aggr="mean", GIN_aggr="add"):
        # No need for bias in GAT Convs due to batch norms
        super(Two_Track_GIN_GAT_Extra_BN, self).__init__() 
        self.preprocess1 = nn.Linear(input_dim, 48, bias=False)
        self.pre_BN1 = BatchNorm(48)
        self.preprocess2 = nn.Linear(48, 64, bias=False)
        self.pre_BN2 = BatchNorm(64)
        
        # Left Track
        self.left_GAT_1 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=GAT_aggr)
        self.left_BN1 = BatchNorm(64)
        
        self.left_GAT_2 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=GAT_aggr)
        self.left_BN2 = BatchNorm(64)

        self.left_GAT_3 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=GAT_aggr)
        self.left_BN3 = BatchNorm(64)

        self.left_GAT_4 = GATv2Conv(64, 8, heads=8, bias=False, dropout=drop_prob, aggr=GAT_aggr)
        self.left_BN4 = BatchNorm(64)
         
        # Right Track
        self.right_GAT_1 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
        self.right_BN1 = BatchNorm(64)
        
        self.right_GAT_2 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
        self.right_BN2 = BatchNorm(64)

        self.right_GAT_3 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
        self.right_BN3 = BatchNorm(64)

        self.right_GAT_4 = GINConv(MLP(3, 64, 64, 64), aggr=GIN_aggr)
        self.right_BN4 = BatchNorm(64)

        self.post_BN = BatchNorm(512)
        self.postprocess1 = nn.Linear(512, 256)
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
        
        combined = self.post_BN(torch.cat((left_block_4_out,right_block_4_out,left_block_3_out,right_block_3_out,left_block_2_out,right_block_2_out,left_block_1_out,right_block_1_out), dim=-1))
        
        x = self.elu(self.postprocess1(combined))
        x = self.elu(self.postprocess2(x))
        x = self.elu(self.postprocess3(x))
        x = self.elu(self.postprocess4(x))
        x = self.elu(self.postprocess5(x))
        x = self.postprocess6(x)

        return x
    
class Two_Track_GAT_GAT(nn.Module):
    def __init__(self, input_dim, output_dim=2, drop_prob=0.1, left_aggr="mean", right_aggr="add"):
        # No need for bias in GAT Convs due to batch norms
        super(Two_Track_GAT_GAT, self).__init__() 
        self.preprocess1 = nn.Linear(input_dim, 48, bias=False)
        self.pre_BN1 = BatchNorm(48)
        self.preprocess2 = nn.Linear(48, 64, bias=False)
        self.pre_BN2 = BatchNorm(64)
        
        # Left Track
        self.left_GAT_1 = GINConv(MLP(3, 64, 64, 64), aggr=left_aggr)
        self.left_BN1 = BatchNorm(64)
        
        self.left_GAT_2 = GINConv(MLP(3, 64, 64, 64), aggr=left_aggr)
        self.left_BN2 = BatchNorm(64)

        self.left_GAT_3 = GINConv(MLP(3, 64, 64, 64), aggr=left_aggr)
        self.left_BN3 = BatchNorm(64)

        self.left_GAT_4 = GINConv(MLP(3, 64, 64, 64), aggr=left_aggr)
        self.left_BN4 = BatchNorm(64)
         
        # Right Track
        self.right_GAT_1 = GINConv(MLP(3, 64, 64, 64), aggr=right_aggr)
        self.right_BN1 = BatchNorm(64)
        
        self.right_GAT_2 = GINConv(MLP(3, 64, 64, 64), aggr=right_aggr)
        self.right_BN2 = BatchNorm(64)

        self.right_GAT_3 = GINConv(MLP(3, 64, 64, 64), aggr=right_aggr)
        self.right_BN3 = BatchNorm(64)

        self.right_GAT_4 = GINConv(MLP(3, 64, 64, 64), aggr=right_aggr)
        self.right_BN4 = BatchNorm(64)

        self.postprocess1 = nn.Linear(512, 256)
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
        
        combined = torch.cat((left_block_4_out,right_block_4_out,left_block_3_out,right_block_3_out,left_block_2_out,right_block_2_out,left_block_1_out,right_block_1_out), dim=-1)
        
        x = self.elu(self.postprocess1(combined))
        x = self.elu(self.postprocess2(x))
        x = self.elu(self.postprocess3(x))
        x = self.elu(self.postprocess4(x))
        x = self.elu(self.postprocess5(x))
        x = self.postprocess6(x)

        return x
