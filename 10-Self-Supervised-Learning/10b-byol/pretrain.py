import torch
import torch.nn.functional as F
from torch import optim

import data
import models
from utils import tqdm


def negative_cosine_similarity(p, z):
    """
    Computes the BYOL negative cosine similarity.

    Args:
        p (Tensor): Online prediction of shape (B, D).
        z (Tensor): Target projection of shape (B, D).

    Returns:
        loss (Tensor): Scalar tensor.
    """

    ################################################################
    # TODO
    
    p = F.normalize(p, dim=-1)
    z = F.normalize(z.detach(), dim=-1)
    cos = torch.sum((p * z), dim=-1)
    loss = 2 - 2 * cos
    loss = loss.mean()
    ################################################################

    return loss


def update_moving_average(online_module, target_module, tau):
    """
    Updates target parameters using an exponential moving average.

    target <- tau * target + (1 - tau) * online
    """

    ################################################################
    # TODO
    with torch.no_grad():
        for op, tp in zip(online_module.parameters(), target_module.parameters()):
            tp.mul_(tau).add_(op, alpha = 1 - tau)
        for ob, tb in zip(online_module.buffers(), target_module.buffers()):
            if torch.is_floating_point(ob):
                tb.mul_(tau).add_(ob, alpha = 1 - tau)
            else:
                tb.copy_(ob)
    ################################################################


def byol_training_step(model, optimizer, x1, x2, device, tau):
    """
    Performs one BYOL training step.
    """

    ################################################################
    # TODO

    x1 = x1.to(device)
    x2 = x2.to(device)

    _, _, p1 = model.online_forward(x1)
    _, _, p2 = model.online_forward(x2)

    with torch.no_grad():
        _, z1 = model.target_forward(x1)
        _, z2 = model.target_forward(x2)

    loss = .5 * (negative_cosine_similarity(p1, z2)+ negative_cosine_similarity(p2, z1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    update_moving_average(model.online_encoder, model.target_encoder, tau)
    update_moving_average(model.online_projector, model.target_projector, tau)

    loss_value = loss.item()
    
    ################################################################

    return loss_value


def train_byol(
    batch_size,
    lr,
    weight_decay,
    epochs,
    device,
    tau,
    num_workers=0,
):
    """
    Pretrains an encoder with BYOL on CIFAR-10.
    """

    ################################################################
    # TODO

    train_loader = data.make_byol_loader(batch_size=batch_size, num_workers=num_workers)
    model = models.BYOL()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loss_history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_samples = 0

        for (x1, x2), _ in train_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            loss_value = byol_training_step(model, optimizer, x1, x2, device, tau)
            epoch_loss += loss_value * x1.shape[0]
            num_samples += x1.shape[0]

        train_loss_history.append(epoch_loss / num_samples)

    ################################################################

    return model, train_loss_history


@torch.no_grad()
def representation_std(model, data_loader, device, max_batches=10):
    """
    Estimates the mean feature standard deviation over a few batches.

    Very small values can indicate representation collapse. This function is
    provided as a diagnostic helper, not as a student task.
    """

    model.eval()
    features = []

    for batch_index, (x, _) in enumerate(data_loader):
        if batch_index >= max_batches:
            break
        x = x.to(device)
        features.append(model(x).detach().cpu())

    features = torch.cat(features, dim=0)
    std = features.std(dim=0).mean().item()

    return std
