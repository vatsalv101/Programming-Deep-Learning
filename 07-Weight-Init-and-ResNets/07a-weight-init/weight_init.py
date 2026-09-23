import torch


def linear_layer_size(weight):
    """
    Returns the n_in and n_out for a linear layer given its weight matrix.

    Args:
        weight (torch.Tensor): The weight matrix of the linear layer with shape (out_features, in_features).

    Returns:
        tuple: A tuple containing n_in and n_out.
    """

    ################################################
    # TODO
    n_out, n_in = weight.shape
    ################################################

    return n_in, n_out


def conv2d_layer_size(weight):
    """
    Returns n_in and n_out for a 2D convolutional layer given its kernel.

    Args:
        weight (torch.Tensor): The kernel of the convolutional layer with shape (C_out, C_in, K_h, K_w).

    Returns:
        tuple: A tuple containing n_in and n_out.
    """

    ################################################
    # TODO
    no, ni, h, w = weight.shape
    n_in = ni * h * w
    n_out = no * h * w
    ################################################

    return n_in, n_out


def lecun_init_(weight, n_in, n_out):
    """
    Initializes the weights of a layer using LeCun initialization.

    The weights are modified in-place.

    Args:
        weight (torch.Tensor): The weight tensor to be initialized.
        n_in (int): The effective input size of the layer.
        n_out (int): The effective output size of the layer.
    """

    ################################################
    with torch.no_grad():
        std = (1 / n_in) ** .5
        weight.normal_(std=std)
    ################################################


def xavier_init_(weight, n_in, n_out):
    """
    Initializes the weights of a layer using Xavier / Glorot initialization.

    The weights are modified in-place.

    Args:
        weight (torch.Tensor): The weight tensor to be initialized.
        n_in (int): The effective input size of the layer.
        n_out (int): The effective output size of the layer.
    """
    
    ################################################
    # TODO
    with torch.no_grad():
        std = (2 / (n_in + n_out)) ** .5
        weight.normal_(std=std)


def kaiming_he_init_(weight, n_in, n_out):
    """
    Initializes the weights of a layer using Kaiming He initialization.

    The weights are modified in-place.

    Args:
        weight (torch.Tensor): The weight tensor to be initialized.
        n_in (int): The effective input size of the layer.
        n_out (int): The effective output size of the layer.
    """

    ################################################
    # TODO
    with torch.no_grad():
        std = (2 / n_in ) ** .5
        weight.normal_(std=std)
    ################################################
