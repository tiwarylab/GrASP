import torch
from model import GAT_model
from torch_geometric.nn import GATConv, GATv2Conv

def distance_sigmoid(data, midpoint, slope):
    """Calculate the distance-based sigmoid function.

    This function takes a distance-based input data and computes the sigmoid value for each element in the input.
    The sigmoid value is calculated using the formula: sigmoid(x) = 1 / (1 + exp(-x)), where x = -slope * (data - midpoint).

    Parameters
    ----------
    data : torch.Tensor
        A tensor containing the input data representing distances.
    midpoint : float
        The midpoint parameter for the sigmoid function. It shifts the sigmoid curve horizontally.
    slope : float
        The slope parameter for the sigmoid function. It controls the steepness of the sigmoid curve.

    Returns
    -------
    torch.Tensor
        A tensor containing the sigmoid values corresponding to the input distances.
    """
    x = -slope*(data-midpoint)
    sigmoid = torch.sigmoid(x)
    
    return sigmoid

def initialize_model(parser_args):
    """Initialize a graph neural network model based on the provided arguments.

    This function creates a graph neural network (GNN) model based on the specified model name and its associated parameters.
    Two differing GNN are supported: GAT and GATv2.
    The model architecture and hyperparameters are determined by the input arguments.

    Parameters
    ----------
    parser_args : argparse.Namespace
        An object containing the parsed command-line arguments. It should include the following attributes:
        - model : str
            The name of the model ('gat' for GAT or 'gatv2' for GATv2).
        - weight_groups : int
            The number of weight groups used in the GNN model.
        - group_layers : int
            The number of layers in each weight group of the GNN model.
        - aggregator : str
            The type of aggregator used in the GNN model.

    Returns
    -------
    torch.nn.Module
        The initialized GNN model as a torch.nn.Module subclass.

    Raises
    ------
    ValueError
        If an unknown model type is specified (i.e., not 'gat' or 'gatv2').
    """
    model_name = parser_args.model
    weight_groups = parser_args.weight_groups
    group_layers = parser_args.group_layers
    aggr = parser_args.aggregator

    if model_name == 'gat':
        print("Using GAT")
        model = GAT_model(input_dim=60,
                          GAT_heads=4, 
                          GAT_style=GATConv,
                          weight_groups=weight_groups, 
                          group_layers=group_layers, 
                          GAT_aggr=aggr)
    elif model_name == 'gatv2':
        print("Using GATv2")
        model = GAT_model(input_dim=60,
                          GAT_heads=4, 
                          GAT_style=GATv2Conv,
                          weight_groups=weight_groups,
                          group_layers=group_layers, 
                          GAT_aggr=aggr)
    else:
        raise ValueError("Unknown Model Type:", model_name)
    return model
