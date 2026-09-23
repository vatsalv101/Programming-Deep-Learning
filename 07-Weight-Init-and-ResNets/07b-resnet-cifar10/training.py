import torch
from torch import nn

from data import make_cifar10_loaders
from resnet import ResNet
from utils import tqdm


def set_learning_rate(optimizer, lr):

    """
    Sets the learning rate of all parameter groups in an optimizer.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer whose learning rate should be changed.
        lr (float): New learning rate.
    """

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def step_learning_rate(epoch, base_lr, milestones=(20, 30), gamma=0.1):

    """
    Computes a step-decay learning rate.

    Args:
        epoch (int): Current epoch index, starting at 0.
        base_lr (float): Learning rate before any decay.
        milestones (tuple[int, ...]): Epoch indices at which the learning rate is multiplied by gamma.
        gamma (float): Multiplicative decay factor.

    Returns:
        lr (float): Learning rate for the current epoch.
    """

    ################################################################
    # TODO

    p = sum(epoch >= milestone for milestone in milestones)
    lr = base_lr * (gamma ** p)
    

    ################################################################

    return lr


def initialize(lr, momentum, weight_decay, device, **model_kwargs):

    """
    Initializes the model and optimizer.

    Args:
        lr (float): Initial learning rate.
        momentum (float): Momentum used by the optimizer.
        weight_decay (float): Weight decay used by the optimizer.
        device (torch.device): Device on which the model should live.
        **model_kwargs: Keyword arguments passed to ResNet.

    Returns:
        model (ResNet): Initialized CIFAR-style ResNet.
        optimizer (torch.optim.Optimizer): Optimizer connected to the model
            parameters.
    """

    ################################################################
    # TODO

    model = ResNet(**model_kwargs).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, weight_decay = weight_decay, momentum = momentum)
        

    ################################################################

    return model, optimizer


def training_step(model, optimizer, loss_fn, x, y, device):

    """
    Performs one optimization step on one batch.

    Args:
        model (nn.Module): Model to train.
        optimizer (torch.optim.Optimizer): Optimizer used for the parameter update.
        loss_fn (nn.Module): Loss function.
        x (torch.Tensor): Input batch.
        y (torch.Tensor): Target labels.
        device (torch.device): Device used for the computation.

    Returns:
        loss_value (float): Loss value for the batch.
    """

    ################################################################
    # TODO

    model.train()

    x = x.to(device)
    y = y.to(device)

    logits = model(x)
    loss = loss_fn(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_value = loss.item()    

    ################################################################

    return loss_value


def train_one_epoch(model, train_loader, optimizer, loss_fn, device):

    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): Model to train.
        train_loader (torch.utils.data.DataLoader): Data loader for training batches.
        optimizer (torch.optim.Optimizer): Optimizer used for parameter updates.
        loss_fn (nn.Module): Loss function.
        device (torch.device): Device used for the computation.

    Returns:
        mean_loss (float): Mean training loss over all batches.
    """

    ################################################################
    # TODO
    loss = 0
    for x, y in train_loader:
        loss += training_step(model, optimizer, loss_fn, x, y, device)
    mean_loss = loss / len(train_loader)   

    ################################################################

    return mean_loss


def evaluate_accuracy(model, data_loader, device):

    """
    Evaluates classification accuracy.

    Args:
        model (nn.Module): Model to evaluate.
        data_loader (torch.utils.data.DataLoader): Data loader for evaluation
            batches.
        device (torch.device): Device used for the computation.

    Returns:
        accuracy (float): Fraction of correctly classified samples.
    """

    ################################################################
    # TODO

    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            yp = model.predict_class(x)
            correct += (y == yp).sum().item()
            total += y.numel()

    accuracy = correct / total
    ################################################################

    return accuracy


def train(
    lr=0.1,
    batch_size=128,
    momentum=0.9,
    weight_decay=0.0001,
    epochs=45,
    milestones=(35, 40),
    shortcut="projection",
    num_workers=0,
):
    """
    Trains a CIFAR-style ResNet on CIFAR-10.

    Args:
        lr (float): Initial learning rate.
        batch_size (int): Number of examples per batch.
        momentum (float): Momentum used by the optimizer.
        weight_decay (float): Weight decay used by the optimizer.
        epochs (int): Number of training epochs.
        milestones (tuple[int, ...]): Epochs at which to decay the learning
            rate.
        shortcut (str): Shortcut type used by the residual blocks.
        num_workers (int): Number of worker processes for data loading.

    Returns:
        model (ResNet): Trained model.
        train_loss_history (list[float]): Mean training loss per epoch.
        test_acc_history (list[float]): Test accuracy per epoch.
    """

    train_loader, test_loader = make_cifar10_loaders(
        batch_size=batch_size,
        num_workers=num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, optimizer = initialize(
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        device=device,
        shortcut=shortcut,
    )
    loss_fn = nn.CrossEntropyLoss()

    train_loss_history = []
    test_acc_history = []

    pbar = tqdm.tqdm(
        range(epochs),
        desc="Train loss: -.---, Test accuracy: -.---",
    )

    ################################################################
    # TODO

    # Basic loop structure is given.
    for epoch in pbar:
        
        lr_epoch = step_learning_rate(epoch, lr, milestones)
        set_learning_rate(optimizer, lr_epoch)

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        test_acc = evaluate_accuracy(model, test_loader, device)
        # Logging
        train_loss_history.append(train_loss)
        test_acc_history.append(test_acc)

        pbar.set_description(
            f"LR: {lr_epoch:.4f}, "
            f"Train loss: {train_loss:.3f}, "
            f"Test accuracy: {100 * test_acc:.2f} %"
        )

    ################################################################

    return model, train_loss_history, test_acc_history
