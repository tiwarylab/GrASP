import torch

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