import torch
from torch import nn, optim

import data
import models
from utils import tqdm


def apply_rotations(x, y):
    """
    Applies rotations to a batch of images.

    Args:
        x (Tensor): Image batch of shape (B, C, H, W).
        y (Tensor): Rotation labels of shape (B,), with values in {0, 1, 2, 3}.

    Returns:
        rotated_x (Tensor): Image batch of shape (B, C, H, W) where image i was rotated by y[i] * 90 degrees.
    """

    ################################################################
    # TODO
    
    rotated = x.clone()

    for k in (1, 2, 3):
        mask = k == y
        rotated[mask] = torch.rot90(x[mask], k, dims=(-2, -1))
        

    ################################################################
    return rotated


def make_random_rotations(x):
    """
    Randomly rotates each image in a batch by 0, 90, 180, or 270 degrees.

    Args:
        x (Tensor): Image batch of shape (B, C, H, W).

    Returns:
        rotated_x (Tensor): Rotated image batch of shape (B, C, H, W).
        y (Tensor): Rotation labels of shape (B,), with values in {0, 1, 2, 3}.
    """

    ################################################################
    # TODO

    y = torch.randint(0, 4, (x.shape[0],), device=x.device, dtype=torch.long)
    x = apply_rotations(x, y)
    ################################################################
    return x, y


def make_deterministic_rotations(x):
    """
    Rotates images in a deterministic pattern for stable RotNet evaluation.

    Image 0 is rotated by 0 degrees, image 1 by 90 degrees, image 2 by
    180 degrees, image 3 by 270 degrees, then the pattern repeats.

    Args:
        x (Tensor): Image batch of shape (B, C, H, W).

    Returns:
        rotated_x (Tensor): Rotated image batch of shape (B, C, H, W).
        y (Tensor): Rotation labels of shape (B,), with values in {0, 1, 2, 3}.
    """

    ################################################################
    # TODO
    # base_repeats = x.shape[0] // 4 + 1
    # y = torch.tensor([0, 1, 2, 3]).repeat(base_repeats)

    # y = y[:x.shape[0]]

    y = torch.arange(x.shape[0], device=x.device, dtype=torch.long) % 4
    x = apply_rotations(x, y)

    ################################################################
    return x, y


def rotnet_training_step(model, optimizer, loss_fn, x, device):
    """
    Performs one training step for the RotNet task.

    Args:
        model (nn.Module): Rotation classifier.
        optimizer (Optimizer): Optimizer used to update the model.
        loss_fn (nn.Module): Loss function.
        x (Tensor): Image batch of shape (B, C, H, W).
        device (torch.device): Device used for computation.

    Returns:
        loss_value (float): Scalar loss value.
    """

    ################################################################
    # TODO

    x = x.to(device)

    x, y = make_random_rotations(x)
    logits = model(x)

    loss = loss_fn(logits, y)
    
    
    ################################################################

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_value = loss.item()

    return loss_value


def train_rotnet(batch_size, lr, weight_decay, epochs, device, num_workers=0):
    """
    Pretrain an encoder using the RotNet self-supervised task.

    The model receives randomly rotated CIFAR-10 images and predicts the
    rotation angle. The original CIFAR-10 class labels are ignored.

    Args:
        batch_size (int): Number of images per batch.
        lr (float): Learning rate.
        weight_decay (float): Weight decay used by the optimizer.
        epochs (int): Number of training epochs.
        device (str): Device used for computation.
        num_workers (int): Number of DataLoader worker processes.

    Returns:
        model (RotationClassifier): Trained rotation classifier.
        train_loss_history (list): Training losses.
        test_acc_history (list): Test accuracies.
    """
    train_loader, test_loader = data.make_cifar10_loaders(batch_size=batch_size, num_workers=num_workers)

    model = models.RotationClassifier()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    train_loss_history = []
    test_acc_history = []

    pbar = tqdm.tqdm(range(epochs), desc="Train loss: -.---, Test accuracy: -.---")

    for _ in pbar:
        model.train()

        epoch_loss = 0.0

        for x, _ in train_loader:
            loss_value = rotnet_training_step(model, optimizer, loss_fn, x, device)
            epoch_loss += loss_value

        epoch_loss /= len(train_loader)
        train_loss_history.append(epoch_loss)

        scheduler.step()

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                x, y = make_deterministic_rotations(x)

                logits = model(x)
                predicted_class = torch.argmax(logits, dim=1)

                correct += (predicted_class == y).sum().item()
                total += y.numel()

        test_acc = correct / total
        test_acc_history.append(test_acc)
        pbar.set_description(f"Train loss: {epoch_loss:.3f}, Test accuracy: {100 * test_acc:.2f} %")

    final_acc = test_acc_history[-1]
    print(f"\nFinal test accuracy: {100 * final_acc:.2f} %")

    return model, train_loss_history, test_acc_history
