import math
import torch

def precompute_ddpm_schedule(num_timesteps, beta_start=1e-4, beta_end=2e-2, device=None):
    """
    Precompute the fixed DDPM noise schedule values.
    """
    betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=-1)
    sqrt_alpha_bars = torch.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)
     
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "sqrt_alpha_bars": sqrt_alpha_bars,
        "sqrt_one_minus_alpha_bars": sqrt_one_minus_alpha_bars
    }


def q_sample(x0, t, noise, sqrt_alpha_bars, sqrt_one_minus_alpha_bars):
    """
    Samples x_t from q(x_t | x_0) during training.
    """
    return sqrt_alpha_bars[t].view(-1, 1, 1, 1) * x0 + sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1) * noise


def get_ddim_timesteps(num_timesteps, num_ddim_steps, step_type="linear", device=None):
    """
    Creates a subsequence of timesteps for accelerated DDIM sampling.

    Args:
        num_timesteps (int): Total diffusion steps T (e.g., 1000).
        num_ddim_steps (int): Number of steps to actually sample S (e.g., 50).
        step_type (str): "linear" or "quadratic" spacing.
        device (torch.device, optional): Target device for the tensor.

    Returns:
        timesteps (Tensor): Subsequence of timesteps in DESCENDING order (e.g., [999, 979, ..., 0]).
    """
    timesteps = None
    ################################################################
    # TODO: Implement step subsampling (linear or quadratic)
    # Hint: Use torch.linspace for linear spacing, round to integers, and reverse!

    timesteps = torch.linspace(0, num_timesteps-1, num_ddim_steps, device = device)
    timesteps = timesteps.round().int()    
    timesteps = torch.flip(timesteps, dims=(0,))
    ################################################################
    return timesteps


@torch.inference_mode()
def ddim_p_sample(model, xt, t, t_prev, schedule, eta=0.0):
    """
    Performs one reverse DDIM step: mapping x_t to x_{t_prev}.

    Args:
        model (nn.Module): U-Net predicting noise epsilon_theta(x_t, t).
        xt (Tensor): Current noisy image batch of shape (B, C, H, W).
        t (int): Current scalar timestep index.
        t_prev (int): Previous scalar timestep index in the DDIM sequence (-1 for final step).
        schedule (dict): Precomputed DDPM schedule dictionary.
        eta (float): DDIM stochasticity parameter. 0.0 is deterministic DDIM, 1.0 is DDPM-like.

    Returns:
        x_prev (Tensor): Denoised sample x_{t_prev} of shape (B, C, H, W).
    """
    x_prev = None
    ################################################################

    was_training = model.training
    model.eval()
    
    alpha_bar_t = schedule['alpha_bars'][t]
    if t_prev < 0:
        alpha_bar_t_prev = torch.tensor(1.0, device=xt.device, dtype=xt.dtype)
    else:
        alpha_bar_t_prev = schedule['alpha_bars'][t_prev]

    tv = torch.full((xt.shape[0],), t, device=xt.device, dtype=torch.long)
    eps = model(xt, tv)
    
    x0 = (xt - schedule['sqrt_one_minus_alpha_bars'][t] * eps) / schedule['sqrt_alpha_bars'][t]
    std = eta * torch.sqrt(
        ((1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t)) * 
        (1.0 - (alpha_bar_t / alpha_bar_t_prev))
    )
    dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_t_prev - std ** 2, min=0.0)) * eps
    x_prev = torch.sqrt(alpha_bar_t_prev) * x0 + dir_xt
    
    if eta > 0.0 and t_prev >= 0:
        x_prev = x_prev + std * torch.randn_like(xt) 
        
    model.train(was_training)

    ################################################################
    return x_prev



@torch.inference_mode()
def ddim_sample(model, num_samples, image_size, num_timesteps, num_ddim_steps, device, schedule, eta=0.0):
    """
    Generates images by accelerated DDIM sampling.

    Args:
        model (nn.Module): Trained noise predictor.
        num_samples (int): Number of images to generate.
        image_size (int): Spatial resolution (H, W).
        num_timesteps (int): Total training timesteps T.
        num_ddim_steps (int): Accelerated sampling steps S.
        device (torch.device): Computation device.
        schedule (dict): Precomputed schedule dictionary.
        eta (float): Stochasticity parameter (default 0.0 for deterministic).

    Returns:
        samples (Tensor): Generated batch of shape (num_samples, C, image_size, image_size).
        samples_trajectories (list): List of intermediate Tensors for animation.
    """
    samples, samples_trajectories = None, []
    ################################################################

    xt = torch.randn(num_samples, model.in_channels, image_size, image_size, device=device)
    samples_trajectories.append(xt.cpu())
    timesteps = torch.linspace(0, num_timesteps-1, num_ddim_steps, device = device)
    timesteps = timesteps.round().int()    
    timesteps = torch.flip(timesteps, dims=(0,))

    for i, t in enumerate(timesteps):
        if i < len(timesteps) - 1:
            t_prev = timesteps[i+1]
        else:
            t_prev = -1
        xt = ddim_p_sample(model, xt, t, t_prev, schedule, eta=eta)
        samples_trajectories.append(xt.cpu())
    samples = xt

    ################################################################
    return samples, samples_trajectories