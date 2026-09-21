import torch


def conv2d_forward_4_loops(x, weight, bias, stride=1, padding=0):

    """
    Forward pass of a 2D convolution layer with four nested loops.

    Args:
        x (torch.Tensor): Input tensor of shape (N, C_in, H, W).
        weight (torch.Tensor): Weight tensor of shape (C_out, C_in, K_h, K_w).
        bias (torch.Tensor): Bias tensor of shape (C_out,).
        stride (int): Stride of the convolution.
        padding (int): Padding of the convolution.

    Returns:
        torch.Tensor: Output tensor of shape (N, C_out, H_out, W_out).
    """

    ################################################
    # TODO

    N, C_in, H, W = x.shape
    C_out, _, K_h, K_w = weight.shape

    H_out = (H + 2 * padding - K_h) // stride + 1
    W_out = (W + 2 * padding - K_w) // stride + 1

    x_padded = torch.zeros((N, C_in, H + 2 * padding, W + 2 * padding), dtype = x.dtype)
    x_padded[:, :, padding:H+padding, padding:W+padding] = x

    output = torch.zeros((N, C_out, H_out, W_out), dtype = x.dtype)

    for n in range(N):
        for c in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    hs = i * stride
                    he = hs + K_h
                    ws = j * stride
                    we = ws + K_w

                    patch = x_padded[n, :, hs:he, ws:we]
                    output[n, c, i, j] = torch.sum(patch * weight[c]) + bias[c]
    
    ################################################

    return output


def conv2d_forward_3_loops(x, weight, bias, stride=1, padding=0):

    """
    Forward pass of a 2D convolution layer with three nested loops.

    Do not loop over the output channels.

    Args:
        x (torch.Tensor): Input tensor of shape (N, C_in, H, W).
        weight (torch.Tensor): Weight tensor of shape (C_out, C_in, K_h, K_w).
        bias (torch.Tensor): Bias tensor of shape (C_out,).
        stride (int): Stride of the convolution.
        padding (int): Padding of the convolution.

    Returns:
        torch.Tensor: Output tensor of shape (N, C_out, H_out, W_out).
    """

    ################################################
    # TODO
    N, C_in, H, W = x.shape
    C_out, _, K_h, K_w = weight.shape

    H_out = (H + 2 * padding - K_h) // stride + 1
    W_out = (W + 2 * padding - K_w) // stride + 1

    x_padded = torch.zeros((N, C_in, H + 2 * padding, W + 2 * padding), dtype = x.dtype)
    x_padded[:, :, padding:H+padding, padding:W+padding] = x

    output = torch.zeros((N, C_out, H_out, W_out), dtype = x.dtype)
    
    for n in range(N):
        for i in range(H_out):
            for j in range(W_out):
                hs = i * stride
                he = hs + K_h
                ws = j * stride
                we = ws + K_w

                patch = x_padded[n, :, hs:he, ws:we]
                output[n, :, i, j] = torch.sum((patch.unsqueeze(0) * weight), dim=(1, 2, 3),) + bias

    return output


def conv2d_forward_1_loop(x, weight, bias, stride=1, padding=0):

    """
    Forward pass of a 2D convolution layer with one loop.

    Only loop over the batch dimension.

    Args:
        x (torch.Tensor): Input tensor of shape (N, C_in, H, W).
        weight (torch.Tensor): Weight tensor of shape (C_out, C_in, K_h, K_w).
        bias (torch.Tensor): Bias tensor of shape (C_out,).
        stride (int): Stride of the convolution.
        padding (int): Padding of the convolution.

    Returns:
        torch.Tensor: Output tensor of shape (N, C_out, H_out, W_out).
    """

    ################################################
    # TODO
    weight_flatten = weight.view(C_out, -1)

    row_idx = (
        torch.arange(H_out).repeat_interleave(K_h) + 
        torch.arange(K_h).repeat(H_out)
    )
        
    col_idx = (
        torch.arange(W_out).repeat_interleave(K_w) + 
        torch.arange(K_w).repeat(W_out)
    )

    for n in range(N):
        image = x_padded[n]

        patches = torch.index_select(image, 1, row_idx)
        patches = torch.index_select(patches, 2, col_idx)

        patches = patches.view(C_in, H_out, K_h, W_out, K_w)
        patches = patches.permute(1, 3, 0, 2, 4)
        patches = patches.reshape(H_out * W_out, C_in * K_h * K_w)

        output_flat = torch.matmul(patches, weight_flatten.t()) + bias.unsqueeze(0)    
        output[n] = output_flat.t().reshape(C_out, H_out, W_out)
    ################################################

    return output
