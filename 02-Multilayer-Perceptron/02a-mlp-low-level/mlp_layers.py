import torch


def relu_forward(x):

    """
    Computes the ReLU activation function element-wise.

    Args:
        x (Tensor): Input tensor of any shape.

    Returns:
        Tensor: ReLU output, same shape as x.
    """

    ################################################################
    # TODO
    return x.clamp(min=0)
    ################################################################


def relu_backward(dout, relu_out):

    """
    Computes the gradient of ReLU.

    Args:
        dout     (Tensor): Upstream gradient, same shape as relu_out.
        relu_out (Tensor): Output of relu_forward.

    Returns:
        Tensor: Gradient w.r.t. x.
    """

    ################################################################
    # TODO
    mask = relu_out > 0
    return dout * mask
    ################################################################


def tanh_forward(x):

    """
    Computes the tanh activation function element-wise.

    Args:
        x (Tensor): Input tensor of any shape.

    Returns:
        Tensor: Element-wise tanh(x), same shape as x.
    """

    ################################################################
    # TODO
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
    ################################################################


def tanh_backward(dout, tanh_out):

    """
    Computes the gradient of tanh given its *output* (not raw input).

    Args:
        dout     (Tensor): Upstream gradient, same shape as tanh_out.
        tanh_out (Tensor): Output of tanh_forward (values between -1 and 1).

    Returns:
        Tensor: Gradient of tanh, same shape.
    """

    ################################################################
    # TODO
    return (1 - tanh_out** 2)  * dout
    ################################################################


def linear_forward(x, W, b):

    """
    Computes a fully connected (linear) layer.

    Args:
        x (Tensor): Input of shape (batch_size, in_features).
        W (Tensor): Weight matrix of shape (in_features, out_features).
        b (Tensor): Bias vector of shape (out_features,).

    Returns:
        out   (Tensor): Output of shape (batch_size, out_features).
        cache  (tuple): (x, W, b) stored for backward pass.
    """

    ################################################################
    # TODO
    out = x @ W + b
    cache = (x, W, b)
    ################################################################

    return out, cache

def linear_backward(dout, cache):

    """
    Computes gradients for a fully connected layer.

    Args:
        dout  (Tensor): Upstream gradient of shape (batch_size, out_features).
        cache (tuple):  (x, W, b) from linear_forward.

    Returns:
        dx (Tensor): Gradient w.r.t. x of shape (batch_size, in_features).
        dW (Tensor): Gradient w.r.t. W of shape (in_features, out_features).
        db (Tensor): Gradient w.r.t. b of shape (out_features,).
    """

    ################################################################
    # TODO
    x, W, b = cache
    dx = dout @ W.T
    dW = x.T @ dout
    db = torch.sum(dout, dim=0)
    
    ################################################################
    
    return dx, dW, db

def softmax(logits):

    """
    Computes softmax probabilities from logits.

    Args:
        logits (Tensor): Raw scores of shape (batch_size, num_classes).

    Returns:
        Tensor: Probabilities of shape (batch_size, num_classes), each row sums to 1.
    """

    ################################################################
    # TODO

    return torch.exp(logits) / torch.sum(torch.exp(logits),dim = -1, keepdim=True)
    ################################################################


def cross_entropy_loss(probs, targets):

    """
    Computes cross-entropy loss and gradient w.r.t. pre-softmax logits.

    Args:
        probs   (Tensor): Softmax probabilities of shape (batch_size, num_classes).
        targets (Tensor): Integer class labels of shape (batch_size,).

    Returns:
        loss    (float):  Mean cross-entropy loss over the batch.
        dlogits (Tensor): Gradient w.r.t. the pre-softmax logits (batch_size, num_classes).
                          This already accounts for the softmax derivative.
    """

    N = probs.shape[0]

    ################################################################
    # TODO
    real_probs = probs[torch.arange(N), targets]
    loss = - 1 / N * (torch.sum(torch.log(real_probs)))

    dlogits = probs.clone()
    dlogits[torch.arange(N), targets] -= 1 
    dlogits /= N
    return loss, dlogits    
    ################################################################
