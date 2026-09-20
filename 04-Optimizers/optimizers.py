import torch


def zero_grad(params):

    """
    Sets all existing gradients to zero.

    Args:
        params (List[Tensor]): All model parameters to zero out gradients for.

    Returns:
        None
    """

    ################################################################
    # TODO

    for param in params:
        if param.grad is not None:
            param.grad.zero_()

    ################################################################


def sgd_step(params, state=None, learning_rate=1e-3):

    """
    Performs one SGD update.

    Args:
        params        (List[Tensor]): Model parameters.
        state         (None): Unused for SGD.
        learning_rate (float): Learning rate.

    Returns:
        None: SGD has no optimizer state.
    """

    ################################################################
    # TODO
    
    with torch.no_grad():
        for p in params:
            if p.grad is not None:
                p -= learning_rate * p.grad    

    ################################################################

    return state


def init_momentum(params):

    """
    Initializes velocity tensors for SGD with momentum.

    Args:
        params (List[Tensor]): Model parameters.

    Returns:
        List[Tensor]: Velocity tensors, one per parameter.
    """

    ################################################################
    # TODO

    velocity = [torch.zeros_like(param) for param in params]

    ################################################################

    return velocity


def sgd_momentum_step(params, state, learning_rate=1e-3, momentum=0.9):

    """
    Performs one SGD with momentum update.

    Args:
        params        (List[Tensor]): Model parameters.
        state         (List[Tensor]): Momentum velocity tensors.
        learning_rate (float): Learning rate.
        momentum      (float): Momentum coefficient.

    Returns:
        List[Tensor]: Updated velocity tensors.
    """
    velocity = state

    ################################################################
    # TODO

    with torch.no_grad():
        for i, param in enumerate(params):
            if param.grad is not None:
                velocity[i] = momentum * velocity[i] - learning_rate * param.grad
                param += velocity[i]
    
    ################################################################

    return velocity


def init_adagrad(params):

    """
    Initializes accumulated squared gradients for AdaGrad.

    Args:
        params (List[Tensor]): Model parameters.

    Returns:
        List[Tensor]: Accumulated squared gradients, one per parameter.
    """

    ################################################################
    # TODO
    
    G = [torch.zeros_like(param) for param in params]

    ################################################################

    return G


def adagrad_step(params, state, learning_rate=1e-2, eps=1e-8):

    """
    Performs one AdaGrad update.

    Args:
        params        (List[Tensor]): Model parameters.
        state         (List[Tensor]): Accumulated squared gradients.
        learning_rate (float): Learning rate.
        eps           (float): Small numerical stability constant.

    Returns:
        List[Tensor]: Updated accumulated squared gradients.
    """
    G = state

    ################################################################
    # TODO

    with torch.no_grad():
        for i, param in enumerate(params):
            if param.grad is not None:
                G[i] += param.grad ** 2
                param -= learning_rate * param.grad / (torch.sqrt(G[i]) + eps)

    
    ################################################################

    return G


def init_rmsprop(params):

    """
    Initializes moving averages of squared gradients for RMSProp.

    Args:
        params (List[Tensor]): Model parameters.

    Returns:
        List[Tensor]: Moving averages of squared gradients, one per parameter.
    """

    ################################################################
    # TODO

    G = [torch.zeros_like(param) for param in params]
        
    ################################################################

    return G


def rmsprop_step(
    params,
    state,
    learning_rate=1e-3,
    gamma=0.9,
    eps=1e-8
):

    """
    Performs one RMSProp update.

    Args:
        params        (List[Tensor]): Model parameters.
        state         (List[Tensor]): Moving averages of squared gradients.
        learning_rate (float): Learning rate.
        gamma         (float): Decay factor for squared gradients.
        eps           (float): Small numerical stability constant.

    Returns:
        List[Tensor]: Updated moving averages of squared gradients.
    """
    G = state

    ################################################################
    # TODO
    
    with torch.no_grad():
        for i, param in enumerate(params):
            if param.grad is not None:
                G[i] = gamma * G[i] + (1 - gamma) * param.grad ** 2

                param -= learning_rate * param.grad / (torch.sqrt(G[i]) + eps)
    ################################################################

    return G


def init_adam(params):

    """
    Initializes Adam first and second moment estimates.

    Args:
        params (List[Tensor]): Model parameters.

    Returns:
        Tuple[List[Tensor], List[Tensor], int]: m, v, t
    """

    ################################################################
    # TODO
    m = [torch.zeros_like(param) for param in params]
    v = [torch.zeros_like(param) for param in params]
    t = 0
    ################################################################

    return m, v, t


def adam_step(
    params,
    state,
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8
):

    """
    Performs one Adam update.

    Args:
        params        (List[Tensor]): Model parameters.
        state         (Tuple): m, v, t.
        learning_rate (float): Learning rate.
        beta1         (float): First moment decay.
        beta2         (float): Second moment decay.
        eps           (float): Small numerical stability constant.

    Returns:
        Tuple[List[Tensor], List[Tensor], int]: Updated m, v, t.
    """
    m, v, t = state

    ################################################################
    # TODO
    t += 1

    with torch.no_grad():
        for i, param in enumerate(params):
            if param.grad is not None:
                m[i] = beta1 * m[i] + (1 - beta1) * param.grad
                v[i] = beta2 * v[i] + (1 - beta2) * param.grad ** 2

                mc = m[i] / (1 - beta1 ** t)
                vc = v[i] / (1 - beta2 ** t)

                param -= learning_rate * mc / (torch.sqrt(vc) + eps)
    ################################################################

    return m, v, t


def adamw_step(
    params,
    state,
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    weight_decay=1e-2
):

    """
    Performs one AdamW update with decoupled weight decay.

    Args:
        params        (List[Tensor]): Model parameters.
        state         (Tuple): m, v, t.
        learning_rate (float): Learning rate.
        beta1         (float): First moment decay.
        beta2         (float): Second moment decay.
        eps           (float): Small numerical stability constant.
        weight_decay  (float): Decoupled weight decay coefficient.

    Returns:
        Tuple[List[Tensor], List[Tensor], int]: Updated m, v, t.
    """
    m, v, t = state

    ################################################################
    # TODO
    t+=1
    with torch.no_grad():
        for i, param in enumerate(params):
            if param.grad is not None:
                m[i] = beta1 * m[i] + (1 - beta1) * param.grad
                v[i] = beta2 * v[i] + (1 - beta2) * param.grad ** 2

                mc = m[i] / (1 - beta1 ** t)
                vc = v[i] / (1 - beta2 ** t)

                param -= learning_rate * weight_decay * param
                param -= learning_rate * mc / (torch.sqrt(vc) + eps) 
    
    ################################################################

    return m, v, t
