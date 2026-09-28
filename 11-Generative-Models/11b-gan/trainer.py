import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import data
import models
import visual


def gan_criterion(probs, targets):
    """
    Computes the GAN loss for a batch of images.

    Args:
        probs (Tensor): Probabilities of shape (B, 1).
        targets (Tensor): Labels of shape (B, 1).

    Returns:
        loss (Tensor): Scalar loss value.
    """
    loss = None
    ################################################################
    # TODO
    probs = probs.view(-1, 1)
    targets = targets.view(-1, 1).to(probs.device, dtype=probs.dtype)
    loss = F.binary_cross_entropy(probs, targets)
    ################################################################
    return loss


def _step(gen, disc, x, device, optimizers=None):
    """
    Performs one training or evaluation step for the GAN.

    If an optimizer is provided, the model is updated. Otherwise, the step
    is run without gradient computation.

    Args:
        gen (nn.Module): Generator model.
        disc (nn.Module): Discriminator model.
        x (Tensor): Image batch of shape (B, C, H, W).
        device (torch.device): Device used for computation.
        optimizers (tuple, optional): Tuple of optimizers used to update the model.

    Returns:
        metrics (dict): Dictionary containing total loss, generator loss,
            and discriminator loss as scalar values.
    """
    gen_loss, disc_loss, disc_loss_real, disc_loss_fake = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
    ################################################################
    # TODO
    
    is_training = optimizers is not None
    gen.train(is_training)
    disc.train(is_training)

    if optimizers:
        go, do = optimizers

    real_labels = torch.ones(x.shape[0])
    fake_labels = torch.zeros(x.shape[0])
    z = torch.randn(x.shape[0], gen.latent_dim, device=device)

    with torch.set_grad_enabled(is_training):
        x_real = x.to(device)
        for p in gen.parameters():
            p.requires_grad_(False)
        x_fake = gen(z)

        disc_loss_real = gan_criterion(disc(x_real), real_labels)
        disc_loss_fake = gan_criterion(disc(x_fake), fake_labels)
        disc_loss = .5 * (disc_loss_real + disc_loss_fake)

        if is_training:
            do.zero_grad()
            disc_loss.backward()
            do.step()

        for p in gen.parameters():
            p.requires_grad_(True)

        for p in disc.parameters():
            p.requires_grad_(False)

        gen_loss = gan_criterion(disc(gen(z)), real_labels)

        if is_training:
            go.zero_grad()
            gen_loss.backward()
            go.step()

        for p in disc.parameters():
            p.requires_grad_(True)

    ################################################################
    return {
        "gen_loss": gen_loss.item(),
        "disc_loss": disc_loss.item(),
        "disc_loss_real": disc_loss_real.item(),
        "disc_loss_fake": disc_loss_fake.item(),
    }


def _loop(gen, disc, dloader, device, optimizers=None):
    """
    Runs one full training or evaluation epoch.

    Args:
        gen (nn.Module): Generator model.
        disc (nn.Module): Discriminator model.
        dloader (DataLoader): DataLoader providing image batches.
        device (torch.device): Device used for computation.
        optimizers (tuple, optional): Tuple of optimizers used for training. If None,
            the models are evaluated without parameter updates.

    Returns:
        epoch_metrics (dict): Average loss values over the full epoch.
    """

    epoch_gen_loss = 0.0
    epoch_disc_loss = 0.0
    epoch_disc_loss_real = 0.0
    epoch_disc_loss_fake = 0.0
    num_samples = 0
    ################################################################
    # TODO
    for x, _ in dloader:
        batch_size = x.shape[0]
        metrics = _step(gen, disc, x, device, optimizers)

        epoch_gen_loss += metrics["gen_loss"] * batch_size
        epoch_disc_loss += metrics["disc_loss"] * batch_size
        epoch_disc_loss_real += metrics["disc_loss_real"] * batch_size
        epoch_disc_loss_fake += metrics["disc_loss_fake"] * batch_size
        num_samples += batch_size

    epoch_gen_loss /= num_samples
    epoch_disc_loss /= num_samples
    epoch_disc_loss_real /= num_samples
    epoch_disc_loss_fake /= num_samples
    ################################################################
    return {
        "gen_loss": epoch_gen_loss,
        "disc_loss": epoch_disc_loss,
        "disc_loss_real": epoch_disc_loss_real,
        "disc_loss_fake": epoch_disc_loss_fake,
    }


def train_gan(batch_size, latent_dim, gen_lr, disc_lr, betas, epochs, device, image_size, num_workers):
    """
    Trains a generative adversarial network on CIFAR-10 images.

    The generator generates new images while the discriminator learns to distinguish between real and fake images.

    Args:
        batch_size (int): Number of images per batch.
        latent_dim (int): Dimensionality of the latent representation.
        gen_lr (float): Learning rate for the generator.
        disc_lr (float): Learning rate for the discriminator.
        epochs (int): Number of training epochs.
        device (str): Device used for computation.
        image_size (int): Size of the images.
        num_workers (int): Number of DataLoader worker processes.

    Returns:
        gen: Trained generator model.
        disc: Trained discriminator model.
        history (dict): Training and validation losses for each epoch.
    """
    gen, disc, history = None, None, None
    ################################################################
    # TODO
    train_loader, val_loader, _ = data.make_afhq_loaders(batch_size, num_workers, image_size)
    device = torch.device(device) if isinstance(device, str) else device

    gen = models.Generator(latent_dim=latent_dim).to(device)
    disc = models.Discriminator().to(device)

    gen_optim = torch.optim.AdamW(gen.parameters(), lr=gen_lr, betas=betas)
    disc_optim = torch.optim.AdamW(disc.parameters(), lr=disc_lr, betas=betas)

    history = {f"{s}_{k}": [] for s in ['train', 'val'] for k in ['gen_loss', 'disc_loss', 'disc_loss_real', 'disc_loss_fake']}

    for _ in range(epochs):
        tm = _loop(gen, disc, train_loader, device, optimizers=(gen_optim, disc_optim))
        vm = _loop(gen, disc, val_loader, device)

        for key in tm:
            history[f"train_{key}"].append(tm[key])
            history[f"val_{key}"].append(vm[key])
    ################################################################
    return gen, disc, history