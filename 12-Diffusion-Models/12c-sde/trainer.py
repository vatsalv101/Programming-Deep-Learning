import torch
from tqdm.auto import tqdm

import data
import utils
import visual
import models


def train_score_sde(base_channels, time_dim, groups, batch_size, lr, betas, weight_decay, epochs, device, num_workers, image_size, beta_start=1e-4, beta_end=20, eps=1e-5, num_steps=1000):
    """
    Trains a score network with denoising score matching.

    This variant uses a fixed U-Net configuration to keep the public API
    focused on the training hyperparameters.

    Args:
        batch_size (int): Number of images per batch.
        lr (float): Learning rate.
        betas (tuple): Betas for the AdamW optimizer.
        weight_decay (float): Weight decay used by the optimizer.
        epochs (int): Number of training epochs.
        device (str | torch.device): Device used for computation.
        num_workers (int): Number of DataLoader worker processes.
        image_size (int): Size of the input images.
        eps (float): Minimum time value.
        beta_start (float): Starting value for the beta schedule.
        beta_end (float): Ending value for the beta schedule.

    Returns:
        model (models.UNet): Trained score model.
        history (dict): Training and validation losses for each epoch.
    """
    model, history = None, None
    ################################################################
    # TODO
    device = torch.device(device)
    train_loader, val_loader, _ = data.make_afhq_loaders(batch_size=batch_size, num_workers=num_workers, image_size=image_size)

    model = models.UNet(in_channels=3, base_channels=base_channels, time_dim=time_dim, groups=groups).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    
    history = {"train_loss": [], "val_loss": []}
    pbar = tqdm(range(epochs), desc="Training VP-SDE model")
    for epoch in pbar:
        model.train()
        train_loss = 0.0
        train_samples = 0

        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            x = x.to(device, non_blocking=True)
            t = torch.rand(x.shape[0], device=device) * (1.0 - eps) + eps
            noise = torch.randn_like(x)
            _, std = utils.marginal_prob(t, beta_start=beta_start, beta_end=beta_end)
            xt = utils.q_sample(x, t, noise, beta_start=beta_start, beta_end=beta_end)

            score = model(xt, t * 999)
            loss = torch.mean(torch.sum((score * std.view(-1, 1, 1, 1) + noise) ** 2, dim=(1, 2, 3)))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.shape[0]
            train_samples += x.shape[0]

        train_loss /= max(train_samples, 1)

        model.eval()
        val_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device, non_blocking=True)
                t = torch.rand(x.shape[0], device=device) * (1.0 - eps) + eps
                noise = torch.randn_like(x)
                _, std = utils.marginal_prob(t, beta_start=beta_start, beta_end=beta_end)
                xt = utils.q_sample(x, t, noise, beta_start=beta_start, beta_end=beta_end)
                score = model(xt, t * 999)
                loss = torch.mean(torch.sum((score * std.view(-1, 1, 1, 1) + noise) ** 2, dim=(1, 2, 3)))

                val_loss += loss.item() * x.shape[0]
                val_samples += x.shape[0]

        val_loss /= max(val_samples, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        pbar.set_postfix({"train_loss": f"{train_loss:.4f}", "val_loss": f"{val_loss:.4f}"})

        if (epoch + 1) % 25 == 0 or epoch + 1 == epochs:
            tqdm.write(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

            samples, _ = utils.vp_sde_sample(model, 16, image_size, num_steps, eps, device, beta_start=beta_start, beta_end=beta_end)

            visual.show_images(samples.cpu(), title="VP-SDE Samples", cols=4).show()
    ################################################################
    return model, history
