import torch


def dropout_forward_train(x, p):
    
    """
    Computes the training-mode forward pass of dropout.

    Args:
        x (torch.Tensor): Input tensor of any shape.
        p (float): Probability of dropping an activation.

    Returns:
        tuple[torch.Tensor, tuple]: Output tensor and cache for the backward pass.
            out (torch.Tensor): Output tensor of the same shape as x.
            cache (tuple): Cache containing the dropout mask and a boolean indicating training mode or evaluation mode.
                mask (torch.Tensor): Dropout mask applied to the input.
                training (bool): True if in training mode, False if in evaluation mode.
    """

    ################################################
    # TODO
    keep = 1 - p
    mask = (torch.randn_like(x) < keep) / keep
    out = x * mask
    cache = (mask, True)

    ################################################

    return out, cache


def dropout_forward_eval(x, p):
    
    """
    Computes the evaluation-mode forward pass of dropout.

    Args:
        x (torch.Tensor): Input tensor of any shape.
        p (float): Probability of dropping an activation during training.

    Returns:
        tuple[torch.Tensor, tuple]: Output tensor and cache for the backward pass.
    
    """
    ################################################
    # TODO
    
    out = x
    cache = (None, False)

    ################################################

    return out, cache


def dropout_backward(dout, cache):
    
    """
    Computes the backward pass of dropout.

    Args:
        dout (torch.Tensor): Upstream gradient of the same shape as the input.
        cache (tuple): Cache returned by dropout_forward_train or dropout_forward_eval.

    Returns:
        torch.Tensor: Gradient with respect to the input.
    """

    ################################################
    # TODO
    
    mask, train = cache
    if train:
        dx = dout * mask
    else:
        dx = dout
    ################################################

    return dx


def maxpool2d_forward(x, kernel_size=2, stride=2):
    
    """
    Computes the forward pass of a 2D max-pooling layer.

    Args:
        x (torch.Tensor): Input tensor of shape (N, C, H, W).
        kernel_size (int): Height and width of each pooling window.
        stride (int): Stride of the pooling operation.

    Returns:
        tuple[torch.Tensor, tuple]: Output tensor and cache (figure out what you need) for the backward pass.
    """

    ################################################
    # TODO
    N, C, H, W = x.shape

    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1

    out = torch.zeros((N, C, H_out, W_out), dtype = x.dtype)

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    hs = i * stride
                    he = hs + kernel_size
                    ws = j * stride
                    we = ws + kernel_size

                    patch = x[n, c, hs:he, ws:we]
                    out[n, c, i, j] = torch.max(patch)

    cache = (out, x, kernel_size, stride)

    ################################################

    return out, cache


def maxpool2d_backward(dout, cache):
    
    """
    Computes the backward pass of a 2D max-pooling layer.

    Args:
        dout (torch.Tensor): Upstream gradient of shape (N, C, H_out, W_out).
        cache (tuple): Cache returned by maxpool2d_forward.

    Returns:
        torch.Tensor: Gradient with respect to the input x.
    """

    ################################################
    # TODO
    out, x, kernel_size, stride = cache
    N, C, H, W = x.shape
    _, _, H_out, W_out = out.shape
    
    dx = torch.zeros_like(x)

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    hs = i * stride
                    he = hs + kernel_size
                    ws = j * stride
                    we = ws + kernel_size

                    patch = x[n, c, hs:he, ws:we]
                    mx = out[n, c, i, j]
                    mask = patch == mx
                    mn = mask.sum()

                    dx[n, c, hs:he, ws:we] += dout[n, c, i, j] * mask / mn

    cache = (out, x, kernel_size, stride)
    
    ################################################

    return dx
