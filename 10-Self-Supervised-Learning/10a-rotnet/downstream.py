import torch
from torch import nn, optim

import data
import models
from utils import tqdm


def create_downstream_model_and_optimizer(rotnet, lr, weight_decay):
    """
    Creates a DownstreamClassifier model and its optimizer.

    Args:
        rotnet (RotationClassifier): Trained rotation classifier.
        lr (float): Learning rate.
        weight_decay (float): Weight decay used by the optimizer.

    Returns:
        model (DownstreamClassifier): Downstream classifier using the pretrained encoder.
        optimizer (Optimizer): Optimizer for the classification head.
    """

    ################################################################
    # TODO
    encoder =  rotnet.encoder
    model = models.DownstreamClassifier(encoder)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr = lr, weight_decay=weight_decay)
    
    ################################################################

    return model, optimizer


def downstream_training_step(model, optimizer, loss_fn, x, y, device):
    """
    Performs one training step for downstream classification.

    Args:
        model (nn.Module): Downstream classifier.
        optimizer (Optimizer): Optimizer used to update the classification head.
        loss_fn (nn.Module): Loss function.
        x (Tensor): Image batch of shape (B, C, H, W).
        y (Tensor): CIFAR-10 labels of shape (B,).
        device (str): Device used for computation.

    Returns:
        loss_value (float): Scalar loss value.
    """

    x = x.to(device)
    y = y.to(device)

    logits = model(x)
    loss = loss_fn(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_value = loss.item()
    return loss_value


def train_downstream(rotnet, batch_size, lr, weight_decay, epochs, device, num_workers=0):
    """
    Train a linear classifier on top of a frozen pretrained encoder.

    The encoder is not updated. Only the linear head is trained using the
    CIFAR-10 class labels.

    Args:
        rotnet (RotationClassifier): Trained rotation classifier.
        batch_size (int): Number of images per batch.
        lr (float): Learning rate.
        weight_decay (float): Weight decay used by the optimizer.
        epochs (int): Number of training epochs.
        device (str): Device used for computation.
        num_workers (int): Number of DataLoader worker processes.

    Returns:
        model (DownstreamClassifier): Trained downstream classifier.
        train_loss_history (list): Training losses.
        test_acc_history (list): Test accuracies.
    """
    train_loader, test_loader = data.make_cifar10_loaders(batch_size=batch_size, num_workers=num_workers)

    model, optimizer = create_downstream_model_and_optimizer(rotnet, lr, weight_decay)
    model.to(device)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    train_loss_history = []
    test_acc_history = []

    pbar = tqdm.tqdm(range(epochs), desc="Train loss: -.---, Test accuracy: -.---")

    for _ in pbar:
        model.train()

        epoch_loss = 0.0

        for x, y in train_loader:
            loss_value = downstream_training_step(model, optimizer, loss_fn, x, y, device)
            epoch_loss += loss_value

        epoch_loss /= len(train_loader)
        train_loss_history.append(epoch_loss)

        scheduler.step()

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)

                predicted_class = model.predict_class(x)

                correct += (predicted_class == y).sum().item()
                total += y.numel()

        test_acc = correct / total
        test_acc_history.append(test_acc)
        pbar.set_description(f"Train loss: {epoch_loss:.3f}, Test accuracy: {100 * test_acc:.2f} %")

    final_acc = test_acc_history[-1]
    print(f"\nFinal test accuracy: {100 * final_acc:.2f} %")

    return model, train_loss_history, test_acc_history
