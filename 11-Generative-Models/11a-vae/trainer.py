import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import data
import models
import visual


def vae_loss(recon, x, mu, logvar):
    """
    Computes the VAE loss for a batch of images.

    The loss consists of a reconstruction loss and a KL divergence term.
    The reconstruction loss is computed as mean squared error.

    Args:
        recon (Tensor): Reconstructed image batch of shape (B, C, H, W).
        x (Tensor): Original image batch of shape (B, C, H, W).
        mu (Tensor): Mean of the approximate posterior distribution.
        logvar (Tensor): Log-variance of the approximate posterior distribution.

    Returns:
        loss (Tensor): Total VAE loss.
        recon_loss (Tensor): Reconstruction loss.
        kl_loss (Tensor): KL divergence loss.
    """
    loss, recon_loss, kl_loss = None, None, None
    ################################################################
    # TODO
    bs = x.shape[0]
    recon_loss = F.mse_loss(recon, x, reduction="sum") / bs
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / bs
    loss = recon_loss + kl_loss
    ################################################################
    return loss, recon_loss, kl_loss


def _step(model, x, device, optimizer=None):
    """
    Performs one training or evaluation step for the VAE.

    If an optimizer is provided, the model is updated. Otherwise, the step
    is run without gradient computation.

    Args:
        model (nn.Module): VAE model.
        x (Tensor): Image batch of shape (B, C, H, W).
        device (torch.device): Device used for computation.
        optimizer (Optimizer, optional): Optimizer used to update the model.

    Returns:
        metrics (dict): Dictionary containing total loss, reconstruction loss,
            and KL loss as scalar values.
    """
    loss, recon_loss, kl_loss = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
    ################################################################
    # TODO
    
    is_training = optimizer is not None
    model.train(is_training)

    with torch.set_grad_enabled(is_training):
        x = x.to(device)

        recon, mu, logvar = model(x)
        loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar)

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    ################################################################
    return {
        "loss": loss.item(),
        "recon_loss": recon_loss.item(),
        "kl_loss": kl_loss.item(),
    }


def _loop(model, dloader, device, optimizer=None):
    """
    Runs one full training or evaluation epoch.

    Args:
        model (nn.Module): VAE model.
        dloader (DataLoader): DataLoader providing image batches.
        device (torch.device): Device used for computation.
        optimizer (Optimizer, optional): Optimizer used for training. If None,
            the model is evaluated without parameter updates.

    Returns:
        epoch_metrics (dict): Average loss values over the full epoch.
    """

    epoch_loss = 0.0
    epoch_recon_loss = 0.0
    epoch_kl_loss = 0.0
    num_samples = 0
    ################################################################
    # TODO
    
    for x, _ in dloader:
        bs = x.shape[0]
        m = _step(model, x, device, optimizer)

        epoch_loss += m['loss'] * bs
        epoch_recon_loss += m['recon_loss'] * bs
        epoch_kl_loss += m['kl_loss'] * bs
        num_samples += bs

    epoch_loss /= num_samples
    epoch_recon_loss /= num_samples
    epoch_kl_loss /= num_samples

    ################################################################
    return {
        "loss": epoch_loss,
        "recon_loss": epoch_recon_loss,
        "kl_loss": epoch_kl_loss,
    }


def train_vae(batch_size, latent_dim, lr, weight_decay, epochs, device, num_workers, image_size):
    """
    Trains a variational autoencoder on CIFAR-10 images.

    The model reconstructs input images while regularizing the latent space
    using the KL divergence term of the VAE loss.

    Args:
        batch_size (int): Number of images per batch.
        latent_dim (int): Dimensionality of the latent representation.
        lr (float): Learning rate.
        weight_decay (float): Weight decay used by the optimizer.
        epochs (int): Number of training epochs.
        device (str): Device used for computation.
        num_workers (int): Number of DataLoader worker processes.
        image_size (int): Size of the input images (assumed square).

    Returns:
        model (VAE): Trained VAE model.
        history (dict): Training and validation losses for each epoch.
    """
    model, history = None, None
    ################################################################
    # TODO
    
    train_loader, val_loader, _ = data.make_afhq_loaders(batch_size, num_workers, image_size)
    device = torch.device(device) if isinstance(device, str) else device
    model = models.VAE(latent_dim=latent_dim, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {f"{s}_{k}":[ ] for s in ['train', 'val'] for k in ['loss', 'recon_loss', 'kl_loss']}

    for epoch in range(epochs):
        tm = _loop(model, train_loader, device, optimizer)
        vm = _loop(model, val_loader, device)

        for k in tm:
            history['train_' + k].append(tm[k])
            history['val_' + k].append(vm[k])

    ################################################################
    return model, history