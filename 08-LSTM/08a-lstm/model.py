import torch
from lstm_cells import lstm_cell_forward, lstm_cell_backward, one_hot_encode


def init_model(vocab_size, H):

    """
    Initializes LSTM model parameters.

    Args:
        vocab_size (int): Number of unique characters (input/output dimension).
        H          (int): Hidden layer size.

    Returns:
            Tuple[Tensor]: (Wx, Wh, b, Wy, by)
            - Wx (Tensor): Input weight matrix   (4H, vocab_size), ~ N(0, 0.01)
            - Wh (Tensor): Hidden weight matrix  (4H, H),          ~ N(0, 0.01)
            - b  (Tensor): LSTM bias vector      (4H, 1),          zeros
            - Wy (Tensor): Output weight matrix  (vocab_size, H),  ~ N(0, 0.01)
            - by (Tensor): Output bias vector    (vocab_size, 1),  zeros
    """

    ################################################################
    # TODO
    Wx = torch.randn(4*H, vocab_size) * .01
    Wh = torch.randn(4*H, H) * .01
    b = torch.zeros(4*H, 1) 
    Wy = torch.randn(vocab_size, H) * .01
    by = torch.zeros(vocab_size, 1)


    ################################################################

    return Wx, Wh, b, Wy, by


def forward_pass(inputs, Wx, Wh, b, Wy, by, hprev, cprev, vocab_size):

    """
    Unrolls the LSTM over a sequence and computes cross-entropy loss.

    At each timestep t:
      1. One-hot encode input
      2. Run lstm_cell_forward
      3. Compute output logits
      4. Get probabilities p
      5. Cross-entropy loss

    Args:
        inputs     (List[int]): Character indices for the input sequence (length T). Wx, Wh, b, Wy, by: Model parameters (see init_model).
        hprev         (Tensor): Initial hidden state (H, 1).
        cprev         (Tensor): Initial cell state (H, 1).
        vocab_size       (int): Vocabulary size (needed for one-hot encoding).

    Returns:
        loss  (float):  Total cross-entropy loss over all timesteps. Summed, not averaged.
        cache  (dict):  All intermediates needed for backward_pass. E.g.,
                 Keys: 'cell_caches', 'ys', 'probs', 'inputs', 'Wy', 'by', 'T'
    """

    T = len(inputs)
    cell_caches = {}
    ys, probs = {}, {}

    h = hprev.clone()
    c = cprev.clone()
    loss = 0.0

    ################################################################
    # TODO

    for t in range(T):
        x_t = one_hot_encode(inputs[t], vocab_size)
        h, c, cache = lstm_cell_forward(x_t, h, c, Wx, Wh, b)
        cell_caches[t] = cache

        ys[t] = Wy @ h + by
        probs[t] = torch.exp(ys[t]) / torch.exp(ys[t]).sum()

        target = inputs[(t+1)%T]
        loss += - torch.log(probs[t][target, 0]).item()


    ################################################################

    cache = {
        'cell_caches': cell_caches,
        'ys': ys, 'probs': probs,
        'inputs': inputs,
        'Wy': Wy, 'by': by,
        'T': T
    }

    return loss, cache


def backward_pass(targets, cache, Wy, by):

    """
    Computes gradients via backpropagation through time (BPTT).

    Args:
        targets (List[int]): Target character indices (length T).
        cache        (dict): Intermediates from forward_pass.
        Wy         (Tensor): Output weight matrix (vocab_size, H).
        by         (Tensor): Output bias (vocab_size, 1).

    Returns:
        Tuple[Tensor]:
            - dWx    (Tensor): Gradient w.r.t. Wx  (4H, vocab_size)
            - dWh    (Tensor): Gradient w.r.t. Wh  (4H, H)
            - dWy    (Tensor): Gradient w.r.t. Wy  (vocab_size, H)
            - db     (Tensor): Gradient w.r.t. b   (4H, 1)
            - dby    (Tensor): Gradient w.r.t. by  (vocab_size, 1)
            - h_last (Tensor): Last hidden state  (H, 1)
            - c_last (Tensor): Last cell state    (H, 1)
    """

    cell_caches = cache['cell_caches']
    probs = cache['probs']
    T = cache['T']

    # Get dimensions from the first cell cache
    H = cell_caches[0]['h_prev'].shape[0]

    ################################################################
    # TODO
    dWx = torch.zeros(4*H, Wy.shape[0])
    dWh = torch.zeros(4*H, H)
    dWy = torch.zeros_like(Wy)
    db = torch.zeros(4*H, 1)
    dby = torch.zeros_like(by)    
    
    dh_next = torch.zeros(H, 1)
    dc_next = torch.zeros(H, 1)

    for t in reversed(range(T)):
        dy = probs[t].clone()
        dy[targets[t]] -= 1

        #y = Wy @ h + by
        h = cell_caches[t]['o'] * torch.tanh(cell_caches[t]['c'])
        dh = dh_next + Wy.T @ dy
        dWy += dy @ h.T
        dby += dy

        dWxt, dWht, dbt, dh_next, dc_next = lstm_cell_backward(dh, dc_next, cell_caches[t])

        dWx += dWxt
        dWh += dWht
        db += dbt
    
    ################################################################

    h_last = cell_caches[T - 1]['o'] * torch.tanh(cell_caches[T - 1]['c'])
    c_last = cell_caches[T - 1]['c']

    return dWx, dWh, dWy, db, dby, h_last, c_last
