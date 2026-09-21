import torch


def conv2d_forward_4_loops(x, weight, bias, stride=1, padding=0):

    """
    Forward pass of a 2D convolution layer with four nested loops.

    This function is provided as given code in this notebook so that the
    backward pass exercise is self-contained and does not depend on notebook 1.

    Args:
        x (torch.Tensor): Input tensor of shape (N, C_in, H, W).
        weight (torch.Tensor): Weight tensor of shape (C_out, C_in, K_h, K_w).
        bias (torch.Tensor): Bias tensor of shape (C_out,).
        stride (int): Stride of the convolution.
        padding (int): Padding of the convolution.

    Returns:
        tuple[torch.Tensor, tuple]: Output tensor of shape
            (N, C_out, H_out, W_out) and cache
            (x, weight, bias, stride, padding) for the backward pass.
    """

    N, C_in, H, W = x.shape
    C_out, _, K_h, K_w = weight.shape

    H_out = (H + 2 * padding - K_h) // stride + 1
    W_out = (W + 2 * padding - K_w) // stride + 1

    x_padded = torch.zeros(
        (N, C_in, H + 2 * padding, W + 2 * padding),
        dtype=x.dtype,
    )
    x_padded[:, :, padding:padding + H, padding:padding + W] = x

    out = torch.zeros(
        (N, C_out, H_out, W_out),
        dtype=x.dtype,
    )

    for n in range(N):
        for c_out in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + K_h
                    w_start = j * stride
                    w_end = w_start + K_w

                    patch = x_padded[n, :, h_start:h_end, w_start:w_end]
                    out[n, c_out, i, j] = torch.sum(patch * weight[c_out]) + bias[c_out]

    cache = (x, weight, bias, stride, padding)
    return out, cache


def conv2d_backward(dout, cache):

    """
    Backward pass of a 2D convolution layer.

    Args:
        dout (torch.Tensor): Upstream gradient of shape
            (N, C_out, H_out, W_out).
        cache (tuple): Tuple (x, weight, bias, stride, padding) from
            conv2d_forward_4_loops.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            Gradients (dx, dweight, db).
    """

    ################################################
    # TODO
    x, weight, bias, stride, padding = cache
    N, C_in, H, W = x.shape
    C_out, _, K_h, K_w = weight.shape
    _, _, H_out, W_out = dout.shape

    x_padded = torch.zeros(
        (N, C_in, H + 2 * padding, W + 2 * padding),
        dtype=x.dtype,
    )
    x_padded[:, :, padding:padding + H, padding:padding + W] = x

    dx_padded = torch.zeros_like(x_padded)
    dweight = torch.zeros_like(weight)
    db = torch.sum(dout, dim=(0,2,3))

    for n in range(N):
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                h_end = h_start + K_h
                w_start = j * stride
                w_end = w_start + K_w

                patch = x_padded[n, :, h_start:h_end, w_start:w_end]
                dweight += patch.unsqueeze(0) * dout[n, :, i, j].view(-1, 1, 1, 1)
                dx_padded[n, :, h_start:h_end, w_start:w_end] += torch.sum((weight * dout[n, :, i, j].view(-1, 1, 1, 1)), dim = 0)


    dx = dx_padded[:, :, padding:H+padding, padding:W+padding]



    







    
    # x, weight, bias, stride, padding = cache
    # N, C_in, H, W = x.shape
    # C_out, _, K_h, K_w = weight.shape
    # _, _, H_out, W_out = dout.shape

    # db = dout.sum(dim=(0, 2, 3))

    # x_padded = torch.zeros((N, C_in, H + 2 * padding, W + 2 * padding), dtype=x.dtype)
    # x_padded[:, :, padding:padding + H, padding:padding + W] = x

    # dx_padded = torch.zeros_like(x_padded)
    # dweight = torch.zeros_like(weight)

    # for n in range(N):
    #     for c_out in range(C_out):
    #         for i in range(H_out):
    #             for j in range(W_out):
    #                 h_start = i * stride
    #                 h_end = h_start + K_h
    #                 w_start = j * stride
    #                 w_end = w_start + K_w

    #                 patch = x_padded[n, :, h_start:h_end, w_start:w_end]
    #                 grad = dout[n, c_out, i, j]

    #                 dweight[c_out] += grad * patch
    #                 dx_padded[n, :, h_start:h_end, w_start:w_end] += grad * weight[c_out]

    # dx = dx_padded[:, :, padding:padding + H, padding:padding + W]

    ################################################

    return dx, dweight, db
