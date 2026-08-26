import torch
from torchvision import datasets

def compute_accuracy(X, y, params, forward_pass, activation_fn):

    """
    Computes classification accuracy.

    Args:
        X             (Tensor): Input data (N, 784).
        y             (Tensor): True labels (N,).
        params          (dict): Model parameters.
        forward_pass (callable): Forward pass function.
        activation_fn (callable): Activation function (relu_forward or tanh_forward).

    Returns:
        float: Fraction of correctly classified images (0 to 1).
    """

    ################################################################
    # TODO
    logits, _ = forward_pass(X, params, activation_fn)
    yp = torch.argmax(logits, dim=-1)
    return (y == yp).float().mean().item()

    ################################################################


def create_batches(X, y, batch_size):

    """
    Shuffles data and splits it into smaller batches.

    Args:
        X          (Tensor): Input data (N, 784).
        y          (Tensor): Labels (N,).
        batch_size (int):    Number of samples per batch.

    Returns:
        list[tuple]: List of (X_batch, y_batch) tuples.
                     Last batch may be smaller than batch_size.
    """

    ################################################################
    # TODO
    perm = torch.randperm(X.shape[0])

    X = X[perm]
    y = y[perm]

    i = 0
    l = []
    while i < X.shape[0] - batch_size:
        l.append((X[i:i+batch_size], y[i:i+batch_size]))
    return l    

    ################################################################


def sgd_step(params, grads, learning_rate):

    """
    Performs a vanilla SGD parameter update in-place.

    Args:
        params        (dict): Model parameters (e.g. 'W1', 'b1', ...).
        grads         (dict): Gradients (e.g. 'dW1', 'db1', ...).
        learning_rate (float): Learning rate.
    """

    ################################################################
    # TODO

    for key in params:
        params[key] -= learning_rate * grads['d'+key]
        
    ################################################################


def train(X_train, y_train, X_val, y_val, params,
          forward_pass, backward_pass, softmax, cross_entropy_loss,
          activation_fn, activation_backward_fn,
          epochs=20, batch_size=64, learning_rate=0.01):

    """
    Trains the MLP with batch SGD.

    Args:
        X_train, y_train       (Tensor): Training data and labels.
        X_val, y_val           (Tensor): Validation data and labels.
        params                   (dict): Model parameters from init_model.
        forward_pass          (callable): Forward pass function.
        backward_pass         (callable): Backward pass function.
        softmax               (callable): Softmax function.
        cross_entropy_loss    (callable): Loss function.
        activation_fn         (callable): Activation (relu_forward or tanh_forward).
        activation_backward_fn(callable): Activation backward (relu_backward or tanh_backward).
        epochs                    (int): Number of training epochs.
        batch_size                (int): Batch size.
        learning_rate           (float): Learning rate for SGD.

    Returns:
        dict: Training history with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
              Each value is a list with one entry per epoch.
    """

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):

        ################################################################
        train_batch_losses = [] # fill this list with the loss of each batch
        ################################################################

        ################################################################
        # TODO

        batches = create_batches(X_train, y_train, batch_size)

        for x, y in batches:
            train_logits, cache = forward_pass(x, params, activation_fn)
            train_probs = softmax(train_logits)
            train_loss, dlogits = cross_entropy_loss(train_probs, y)
            grads = backward_pass(dlogits, cache, activation_backward_fn)
            sgd_step(params, grads, learning_rate)
            train_batch_losses.append(train_loss)
        
        ################################################################

        # Epoch metrics
        train_loss = sum(train_batch_losses) / len(train_batch_losses)

        val_logits, _ = forward_pass(X_val, params, activation_fn)
        val_probs = softmax(val_logits)
        val_loss, _ = cross_entropy_loss(val_probs, y_val)

        train_acc = compute_accuracy(X_train, y_train, params, forward_pass, activation_fn)
        val_acc = compute_accuracy(X_val, y_val, params, forward_pass, activation_fn)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | train_acc: {train_acc:.2%} | val_acc: {val_acc:.2%}")

    

    return history

# HELPER FUNCTION TO LOAD MNIST
def load_mnist(data_dir='./data'):
    """
    Loads MNIST and returns flat tensors with a train/val/test split.

    Returns:
        Tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
            - X_train (50000, 784), y_train (50000,)
            - X_val   (10000, 784), y_val   (10000,)
            - X_test  (10000, 784), y_test  (10000,)
    """
    train_set = datasets.MNIST(data_dir, train=True, download=True)
    test_set = datasets.MNIST(data_dir, train=False, download=True)

    X_full = train_set.data.float().view(-1, 784) / 255.0
    y_full = train_set.targets

    X_train, X_val = X_full[:50000], X_full[50000:]
    y_train, y_val = y_full[:50000], y_full[50000:]

    X_test = test_set.data.float().view(-1, 784) / 255.0
    y_test = test_set.targets

    return X_train, y_train, X_val, y_val, X_test, y_test