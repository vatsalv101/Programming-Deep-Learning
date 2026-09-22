import torch


def init_layernorm1d(num_features):

    """
    Initializes the tensors for a LayerNorm1d forward pass.

    Args:
        num_features (int): Number of features D.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            gamma, beta, both of shape (D,).
    """
    
    ################################################
    # TODO
    gamma = torch.ones(num_features)
    beta = torch.zeros(num_features)
    ################################################

    return gamma, beta


def layernorm1d_forward(x, gamma, beta, eps=1e-5):

    """
    Computes the forward pass of LayerNorm1d.

    Args:
        x (torch.Tensor): Input tensor of shape (N, D).
        gamma (torch.Tensor): Scale tensor of shape (D,).
        beta (torch.Tensor): Shift tensor of shape (D,).
        eps (float): Small constant for numerical stability.

    Returns:
        torch.Tensor: Output tensor y of shape (N, D).
    """

    ################################################
    # TODO

    m = torch.mean(x, dim = 1)[:, None]
    std = torch.var(x, dim = 1, correction = 0)[:, None]
    xp = (x - m) / torch.sqrt(std + eps)
    y = gamma * xp + beta
   



    
    # mu = torch.mean(x, dim = 1)[:, None]
    # var = torch.var(x, dim = 1, correction=0)[:, None]
    # x = (x-mu) / torch.sqrt(var+eps)
    # y = gamma * x + beta
    ################################################

    return y
