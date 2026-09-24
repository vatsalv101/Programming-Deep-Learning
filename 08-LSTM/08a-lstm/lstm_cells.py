import torch


def one_hot_encode(ix, vocab_size):
    
    """
    Creates a one-hot encoded column vector.

    Args:
        ix (int): Index of the active character.
        vocab_size (int): Length of the vector (vocabulary size).

    Returns:
        Tensor: One-hot vector of shape (vocab_size, 1) with a 1.0 at position ix.
    """

    ################################################################
    # TODO
    oh = torch.zeros(vocab_size, 1)
    oh[ix] = 1
    return oh
    ################################################################


def sigmoid(x):

    """
    Computes the sigmoid activation function element-wise.

    Args:
        x (Tensor): Input tensor of any shape.

    Returns:
        Tensor: Element-wise sigmoid, same shape as x.
    """

    ################################################################
    # TODO
    return 1 / (1 + torch.exp(-x))
    ################################################################


def sigmoid_deriv(s):

    """
    Computes the derivative of sigmoid given its *output* s (not the raw input).

    Args:
        s (Tensor): Sigmoid output values (between 0 and 1).

    Returns:
        Tensor: Derivative values, same shape as s.
    """

    ################################################################
    # TODO
    return s * (1 - s)
    ################################################################


def lstm_cell_forward(x_t, h_prev, c_prev, Wx, Wh, b):

    """
    Performs a single LSTM cell forward step. one timestep.

    Computes gates (f, i, g, o), updates cell state, and produces new hidden state.

    Args:
        x_t    (Tensor): Input vector (one-hot encoded) at this timestep (vocab_size, 1).
        h_prev (Tensor): Previous hidden state (H, 1).
        c_prev (Tensor): Previous cell state (H, 1).
        Wx     (Tensor): Input-to-hidden weight matrix (4H, vocab_size).
        Wh     (Tensor): Hidden-to-hidden weight matrix (4H, H).
        b      (Tensor): Bias vector (4H, 1).

    Returns:
        h     (Tensor): New hidden state (H, 1).
        c     (Tensor): New cell state (H, 1).
        cache (dict):   Intermediate values needed for backward pass. E.g.,
                        Keys: 'x_t', 'h_prev', 'c_prev', 'f', 'i', 'g', 'o', 'c', 'preactivations', 'Wh'
    """

    H = h_prev.shape[0]

    ################################################################
    # TODO
    preactivations = Wx @ x_t + Wh @ h_prev + b

    f = torch.sigmoid(preactivations[0:H])
    i = torch.sigmoid(preactivations[H:2*H])
    g = torch.tanh(preactivations[2*H:3*H])
    o = torch.sigmoid(preactivations[3*H:4*H])

    c= f * c_prev + i * g
    h = o * torch.tanh(c)

    cache = {'x_t':x_t, 'h_prev':h_prev, 'c_prev':c_prev, 'f':f, 'i':i, 'g':g, 'o':o, 'c':c, 'preactivations':preactivations, 'Wh':Wh}
    ################################################################

    return h, c, cache


def lstm_cell_backward(dh_next, dc_next, cache):

    """
    Computes gradients for a single LSTM cell backward step. one timestep.

    Args:
        dh_next (Tensor): Gradient of loss w.r.t. hidden state h from future timestep (H, 1).
        dc_next (Tensor): Gradient of loss w.r.t. cell state c from future timestep (H, 1).
        cache   (dict):   Intermediate values from lstm_cell_forward.

    Returns:
        dWx     (Tensor): Gradient w.r.t. Wx (4H, vocab_size).
        dWh     (Tensor): Gradient w.r.t. Wh (4H, H).
        db      (Tensor): Gradient w.r.t. b  (4H, 1).
        dh_prev (Tensor): Gradient w.r.t. previous hidden state (H, 1).
        dc_prev (Tensor): Gradient w.r.t. previous cell state (H, 1).
    """

    # TODO
    
    x_t = cache['x_t']
    h_prev = cache['h_prev']
    c_prev = cache['c_prev']
    f = cache['f']
    i = cache['i']
    g = cache['g']
    o = cache['o']
    preactivations = cache['preactivations']
    c = cache['c']
    Wh = cache['Wh']
    H = h_prev.shape[0]

    dtmp = torch.zeros_like(preactivations)

    dc = dc_next + o * (1 - torch.tanh(c)**2) * dh_next

    dtmp[0:H] = c_prev * dc * f * (1 - f)
    dtmp[H:2*H] = g * dc * i * (1 - i)
    dtmp[2*H:3*H] = (1 - g**2) * i * dc
    dtmp[3*H:4*H] = torch.tanh(c) * dh_next * o * (1 - o)

    dWx = dtmp @ x_t.T
    dWh = dtmp @ h_prev.T
    db = dtmp
    dh_prev = Wh.T @ dtmp
    dc_prev = f * dc

    

    ################################################################

    return dWx, dWh, db, dh_prev, dc_prev