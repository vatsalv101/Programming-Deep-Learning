import torch
from torch import nn, optim

import data
import models
from utils import tqdm


def train_downstream(
    byol,
    batch_size,
    lr,
    weight_decay,
    epochs,
    device,
    num_workers=0,
):
    """
    Train a linear classifier on top of the frozen BYOL encoder.

    This file intentionally gives only one scaffolded function. Reuse training
    code from previous CIFAR-10 exercises.
    """

    ################################################################
    # TODO

    train_loader, test_loader = data.make_cifar10_loaders(
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = models.LinearClassifier(byol.online_encoder)
    model.to(device)

    for p in model.encoder.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    train_loss_history = []
    test_acc_history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x.shape[0]

        train_loss_history.append(epoch_loss / len(train_loader.dataset))

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

        test_acc_history.append(correct / total)

    ################################################################

    return model, train_loss_history, test_acc_history
