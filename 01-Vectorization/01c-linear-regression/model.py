import torch
from typing import Optional
from torch.utils.data import DataLoader

def mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

    """
    Calculates the mean squared error between predictions and targets (a.k.a. true values).

    Args:
        y_pred (Tensor): predicted values of shape (N, )
        y_true (Tensor): true values of shape (N, )

    Returns:
        mse (Tensor): mean squared error (scalar tensor)
    """

    ################################################################
    # TODO
    mse = torch.mean((y_pred - y_true) ** 2)
    ################################################################
    return mse

def mse_gradients(y_pred: torch.Tensor, y_true: torch.Tensor)->torch.Tensor:

    """
    Calculates the gradient of the mean squared error between y_pred and y_true w.r.t. to y_pred

    Args:
        y_pred (Tensor): predicted values of shape (N, )
        y_true (Tensor): true values of shape (N, )

    Returns:
        grad_mse(Tensor): gradient of the MSE w.r.t. y_pred of shape (N, )
    """
    ################################################################
    # TODO
    grad_mse = 2 / len(y_pred) * (y_pred - y_true)
    ################################################################
    return grad_mse

def linear_predict(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor)->torch.Tensor:

    """
    Calculates the output of a linear prediction model with parameters w and b for a batch of data points X

    Args:
        X (Tensor): data tensor of shape (N, D)
        w (Tensor): weight tensor of shape (D, )
        b (Tensor): bias tensor of shape (, )

    Returns:
        predictions (Tensor): predicted values of shape (N, )
    """
    ################################################################
    # TODO
    predictions = X @ w + b
    ################################################################
    return predictions

def linear_gradients(X: torch.Tensor, y: torch.Tensor, w: torch.Tensor, b: torch.Tensor)->tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    """
    Calculates the gradient of the loss of the linear model for a batch of data w.r.t to the parameters w and b.

    Args:
        X (Tensor): data tensor of shape (N, D)
        y (Tensor): target tensor of shape (N, )
        w (Tensor): weight tensor of shape (D, )
        b (Tensor): bias tensor of shape (, )
    Returns:
        error:  mse between the prediction of the model and y
        grad_w: gradient of w of shape (D, )
        grad_b: gradient of b of shape (, )
    """
    y_prediction = linear_predict(X, w, b)
    error = mse(y_prediction, y)
    grad_mse = mse_gradients(y_prediction, y) # (N, )

    ################################################################
    # TODO

    grad_w = X.T @ grad_mse
    grad_b = torch.sum(grad_mse)
    
    ################################################################
    return error, grad_w, grad_b

def fit(dataloader: DataLoader,
        learning_rate: float,
        epochs: int = 50,
        initial_w: Optional[torch.Tensor] = None,
        initial_b: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, list]:
    
    """
    Fit a linear regression model with gradient descent.

    Args:
        dataloader    (DataLoader): DataLoader that provides batches of (X, y) data for training. X should have shape (N, D) and y should have shape (N, ).
        learning_rate (float): The stepsize for the parameter adjustments.
        epochs        (int): The number of epochs (how often each data point is seen).
        initial_w     (Optional): The weight to start the optimization. If None, weights are initialized randomly.
        initial_b     (Optional): The bias to start the optimization. If None, bias is initialized to zero.

    Returns:
        w (Tensor): the trained weights
        b (Tensor): the trained bias
        err_history (list): a list containing the mean training error of each epoch
    """
    error_history = []

    D = next(iter(dataloader))[0].shape[1] # get next (X, y), then X has shape (N, D)

    if initial_w is None:
        w = torch.randn(D)
    else:
        assert initial_w.shape == (D,)
        w = initial_w

    if initial_b is None:
        b = torch.zeros(1)
    else:
        b = initial_b

    for e in range(epochs):
        epoch_error = 0.0
        for X, y in dataloader:
            error, grad_w, grad_b = linear_gradients(X, y, w, b)
            epoch_error += error

            ################################################################
            # TODO
            # optimize with gradient descent
            w -= learning_rate * grad_w
            b -= learning_rate * grad_b
            ################################################################

        error_history.append(epoch_error / len(dataloader)) # take mean over all batches
    return w, b, error_history
