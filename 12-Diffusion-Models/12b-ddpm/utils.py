import math

import torch

def precompute_ddpm_schedule(num_timesteps, beta_start=1e-4, beta_end=2e-2, device=None):

    """
    Precompute the fixed DDPM noise schedule values following the existing naming scheme.

    Args:
        num_timesteps (int): Number of diffusion steps T.
        beta_start (float): First beta value.
        beta_end (float): Last beta value.
        device (torch.device, optional): Device for the returned tensor.

    Returns:
        schedule (dict): Dictionary containing betas, alphas, alpha_bars,
            sqrt_alpha_bars, sqrt_one_minus_alpha_bars with shapes (T,), (T,), (T,), (T,), (T,) respectively.
    """

    betas, alphas, alpha_bars, sqrt_alpha_bars, sqrt_one_minus_alpha_bars = None, None, None, None, None
    ################################################################
    # TODO

    betas = torch.linspace(beta_start, beta_end, num_timesteps, device = device)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    sqrt_alpha_bars = torch.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars)
    
    ################################################################

    return {"betas": betas,
            "alphas": alphas,
            "alpha_bars": alpha_bars,
            "sqrt_alpha_bars": sqrt_alpha_bars,
            "sqrt_one_minus_alpha_bars": sqrt_one_minus_alpha_bars
            }

def q_sample(x0, t, noise, sqrt_alpha_bars, sqrt_one_minus_alpha_bars):

    """
    Samples x_t from q(x_t | x_0).

    Args:
        x0 (Tensor): Clean image batch of shape (B, C, H, W).
        t (Tensor): Time indices of shape (B,).
        noise (Tensor): Gaussian noise with the same shape as x0.
        sqrt_alpha_bars (Tensor): Square root of cumulative alpha products of shape (T,).
        sqrt_one_minus_alpha_bars (Tensor): Square root of (1 - cumulative alpha products) of shape (T,).

    Returns:
        xt (Tensor): Noised image batch of shape (B, C, H, W).
    """

    xt = None
    ################################################################
    # TODO
    
    xt = sqrt_alpha_bars[t].view(-1, 1, 1, 1) * x0 + sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1) * noise
    ################################################################

    return xt

@torch.inference_mode()
def ddpm_p_sample(model, xt, t, t_index, schedule):

    """
    Performs one reverse DDPM sampling step p_theta(x_{t-1} | x_t).

    Args:
        model (nn.Module): U-Net predicting epsilon_theta(x_t, t).
        xt (Tensor): Current noisy image batch of shape (B, C, H, W).
        t (Tensor): Time indices of shape (B,).
        t_index (int): Current scalar time index.
        schedule (dict): Precomputed DDPM schedule.

    Returns:
        x_prev (Tensor): Previous sample x_{t-1} of shape (B, C, H, W).
    """

    x_prev = None
    ################################################################
    # TODO
    was_training = model.training
    model.eval()

    eps = model(xt, t)
    mu = 1 / torch.sqrt(schedule['alphas'][t_index]) * (xt - (schedule['betas'][t_index] / schedule['sqrt_one_minus_alpha_bars'][t_index] * eps))
    x_prev = mu

    if t_index > 0:
        var = schedule['sqrt_one_minus_alpha_bars'][t_index - 1] / schedule['sqrt_one_minus_alpha_bars'][t_index] * schedule['betas'][t_index]
        x_prev = mu + var * torch.randn_like(var)

    model.train(was_training)
    ################################################################

    return x_prev


@torch.inference_mode()
def ddpm_sample(model, num_samples, image_size, num_timesteps, device, schedule):

    """
    Generates images by ancestral DDPM sampling.

    Args:
        model (nn.Module): Trained DDPM noise predictor.
        num_samples (int): Number of samples to generate.
        image_size (int): Spatial image size.
        num_timesteps (int): Number of reverse diffusion steps.
        device (torch.device): Device used for computation.

    Returns:
        samples (Tensor): Generated image batch of shape (num_samples, 3, image_size, image_size).
        samples_trajectories (list): List of intermediate samples for visualization.
    """

    samples, samples_trajectories = None, []
    ################################################################
    # TODO
    was_training = model.training
    model.eval()

    xt = torch.randn(num_samples, model.in_channels, image_size, image_size)
    samples_trajectories.append(xt.cpu())

    for t in reversed(range(num_timesteps)):
        tv = torch.ones(num_samples) * t
        xt = ddpm_p_sample(model, xt, tv, t, schedule)
        samples_trajectories.append(xt.cpu())
    samples = xt

    model.train(was_training)
    ################################################################
    
    return samples, samples_trajectories