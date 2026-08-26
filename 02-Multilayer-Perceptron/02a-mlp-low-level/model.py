import torch
from mlp_layers import linear_forward, linear_backward


def init_model():

    """
    Initializes MLP parameters for a 784 -> 256 -> 128 -> 64 -> 10 architecture.

    Returns:
        dict: Parameter dictionary with keys:
            - 'W1' (784, 256), 'b1' (256,)
            - 'W2' (256, 128), 'b2' (128,)
            - 'W3' (128, 64),  'b3' (64,)
            - 'W4' (64, 10),   'b4' (10,)
            Weights ~ N(0, 0.1), biases = 0.
    """

    ################################################################
    # TODO
    return {'W1': torch.randn(784, 256) * .1, 'b1':torch.zeros(256,),
            'W2': torch.randn(256, 128) * .1, 'b2':torch.zeros(128,),
            'W3': torch.randn(128, 64) * .1,  'b3':torch.zeros(64,),
            'W4': torch.randn(64, 10) * .1,   'b4':torch.zeros(10,)}

    ################################################################


def forward_pass(X, params, activation_fn):

    """
    MLP forward pass: 3 hidden layers with activation, one output layer (logits).

    Args:
        X             (Tensor): Input of shape (batch_size, 784).
        params          (dict): Parameters from init_model.
        activation_fn (callable): Activation function (relu_forward or tanh_forward).

    Returns:
        logits (Tensor): Output logits of shape (batch_size, 10).
        cache    (dict): All intermediates needed for backward_pass.
    """

    ################################################################
    # TODO

    z1, cache1 = linear_forward(X, params['W1'], params['b1'])  
    a1 = activation_fn(z1)  
    z2, cache2 = linear_forward(a1, params['W2'], params['b2'])  
    a2 = activation_fn(z2)  
    z3, cache3 = linear_forward(a2, params['W3'], params['b3'])  
    a3 = activation_fn(z3)  
    z4, cache4 = linear_forward(a3, params['W4'], params['b4'])  
    
    cache = {
        'cache1': cache1, 'a1': a1,
        'cache2': cache2, 'a2': a2,
        'cache3': cache3, 'a3': a3,
        'cache4': cache4,
    }
    return z4, cache
    ################################################################


def backward_pass(dlogits, cache, activation_backward_fn):

    """
    MLP backward pass: computes gradients for all 4 layers.

    Args:
        dlogits                (Tensor): Gradient w.r.t. logits (batch_size, 10).
        cache                    (dict): Intermediates from forward_pass.
        activation_backward_fn (callable): Backward function (relu_backward or tanh_backward),
                           called with (upstream_gradient, activation_output).

    Returns:
        dict: Gradients with keys 'dW1', 'db1', 'dW2', 'db2', 'dW3', 'db3', 'dW4', 'db4'.
    """

    ################################################################
    # TODO

    da3, dW4, db4 = linear_backward(dlogits, cache['cache4'])

    dz3 = activation_backward_fn(da3, cache['a3'])
    da2, dW3, db3 = linear_backward(dz3, cache['cache3'])

    dz2 = activation_backward_fn(da2, cache['a2'])
    da1, dW2, db2 = linear_backward(dz2, cache['cache2'])

    dz1 = activation_backward_fn(da1, cache['a1'])
    _, dW1, db1 = linear_backward(dz1, cache['cache1'])

    grads = {
        'dW1': dW1, 'db1': db1,
        'dW2': dW2, 'db2': db2,
        'dW3': dW3, 'db3': db3,
        'dW4': dW4, 'db4': db4,
    }
    return grads

    ################################################################
