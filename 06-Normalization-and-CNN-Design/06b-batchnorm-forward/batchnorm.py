import torch


def init_batchnorm2d(num_channels):
    """
    Initializes the tensors for a BatchNorm2d forward pass.

    Args:
        num_channels (int): Number of channels C.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            gamma, beta, running_mean, running_var, all of shape (C,).
    """
    ################################################
    # TODO

    gamma = torch.ones(num_channels)
    beta = torch.zeros(num_channels)
    running_mean = torch.zeros(num_channels)
    running_var = torch.ones(num_channels)
    
    ################################################

    return gamma, beta, running_mean, running_var


def batchnorm2d_forward_train(
    x,
    gamma,
    beta,
    running_mean,
    running_var,
    eps=1e-5,
    momentum=0.1,
):
    """
    Computes the training-mode forward pass of BatchNorm2d.

    Args:
        x (torch.Tensor): Input tensor of shape (N, C, H, W).
        gamma (torch.Tensor): Scale tensor of shape (C,).
        beta (torch.Tensor): Shift tensor of shape (C,).
        running_mean (torch.Tensor): Running mean of shape (C,).
        running_var (torch.Tensor): Running variance of shape (C,).
        eps (float): Small constant for numerical stability.
        momentum (float): Momentum used for updating running statistics.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            Output tensor y of shape (N, C, H, W), updated running_mean,
            and updated running_var.
    """
    ################################################
    # TODO
    alpha = momentum
    mu = torch.mean(x, dim=(0, 2, 3))
    var = torch.var(x, dim=(0, 2, 3), correction = 0)
    varr = torch.var(x, dim=(0, 2, 3), correction = 1)

    xc = (x - mu.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + eps)
    y = xc * gamma.view(1, -1, 1, 1) + beta.view(1, -1, 1, 1)

    new_running_mean = (1 - alpha) * running_mean + alpha * mu
    new_running_var = (1 - alpha) * running_var + alpha * varr













    # batch_mean = torch.mean(x, dim=(0,2,3))
    # batch_var = torch.var(x, dim=(0,2,3), correction = 0)

    # x_h = (x - batch_mean.view(1, -1, 1, 1)) / torch.sqrt(batch_var.view(1, -1, 1, 1) + eps)
    # y = gamma.view(1, -1, 1, 1) * x_h + beta.view(1, -1, 1, 1)

    # batch_running_var = torch.var(x, dim=(0,2,3), correction = 1)
    # new_running_mean = (1 - momentum) * running_mean + momentum * batch_mean
    # new_running_var = (1 - momentum) * running_var + momentum * batch_running_var  
    ################################################

    return y, new_running_mean, new_running_var


def batchnorm2d_forward_eval(
    x,
    gamma,
    beta,
    running_mean,
    running_var,
    eps=1e-5,
):
    """
    Computes the evaluation-mode forward pass of BatchNorm2d.

    Args:
        x (torch.Tensor): Input tensor of shape (N, C, H, W).
        gamma (torch.Tensor): Scale tensor of shape (C,).
        beta (torch.Tensor): Shift tensor of shape (C,).
        running_mean (torch.Tensor): Running mean of shape (C,).
        running_var (torch.Tensor): Running variance of shape (C,).
        eps (float): Small constant for numerical stability.

    Returns:
        torch.Tensor: Output tensor y of shape (N, C, H, W).
    """
    ################################################
    # TODO
    x = (x - running_mean.view(1, -1, 1, 1)) / torch.sqrt(running_var.view(1, -1, 1, 1) + eps)
    y = gamma.view(1, -1, 1, 1) * x + beta.view(1, -1, 1 ,1)






    # x_h = (x - running_mean.view(1, -1, 1, 1)) / torch.sqrt(running_var.view(1, -1, 1, 1)+ eps)
    # y = gamma.view(1, -1, 1, 1) * x_h + beta.view(1, -1, 1, 1)
    ################################################

    return y
