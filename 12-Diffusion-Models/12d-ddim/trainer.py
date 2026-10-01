import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import data
import utils
import models


def train_ddpm(num_timesteps, base_channels, time_dim, groups, batch_size, lr, betas, weight_decay, epochs, device, num_workers, image_size):
    """
    Trains a noise prediction model on AFHQ images. 
    The trained model can be sampled using either DDPM or DDIM!
    """
    device = torch.device(device)
    train_loader, val_loader, _ = data.make_afhq_loaders(
        batch_size=batch_size, num_workers=num_workers, image_size=image_size
    )

    model = models.UNet(in_channels=3, base_channels=base_channels, time_dim=time_dim, groups=groups).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    schedule = utils.precompute_ddpm_schedule(num_timesteps, beta_start=1e-4, beta_end=2e-2, device=device)

    history = {"train_loss": [], "val_loss": []}

    for epoch in tqdm(range(epochs), desc="Epochs"):
        # -- Training --
        model.train()
        train_loss = 0.0
        num_train_samples = 0

        for x, _ in train_loader:
            x = x.to(device)
            t = torch.randint(0, num_timesteps, (x.shape[0],), device=device)
            noise = torch.randn_like(x)
            xt = utils.q_sample(x, t, noise, schedule["sqrt_alpha_bars"], schedule["sqrt_one_minus_alpha_bars"])

            pred_noise = model(xt, t)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.shape[0]
            num_train_samples += x.shape[0]

        train_loss /= max(1, num_train_samples)

        # -- Validation --
        model.eval()
        val_loss = 0.0
        num_val_samples = 0

        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                t = torch.randint(0, num_timesteps, (x.shape[0],), device=device)
                noise = torch.randn_like(x)
                xt = utils.q_sample(x, t, noise, schedule["sqrt_alpha_bars"], schedule["sqrt_one_minus_alpha_bars"])

                pred_noise = model(xt, t)
                loss = F.mse_loss(pred_noise, noise)

                val_loss += loss.item() * x.shape[0]
                num_val_samples += x.shape[0]

        val_loss /= max(1, num_val_samples)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

    return model, history