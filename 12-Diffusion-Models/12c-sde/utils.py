import math
import torch

def vp_coeffs(x, t, beta_start=0.1, beta_end=20):

    """
    Computes the drift and diffusion coefficient for the VP SDE.

    Args:
        x (Tensor): Noised image batch of shape (B, C, H, W).
        t (Tensor): Continuous time tensor of shape (B,).
        beta_start (float): First beta value.
        beta_end (float): Last beta value.

    Returns:
        drift (Tensor): Drift coefficient tensor of shape (B,).
        diffusion (Tensor): Diffusion coefficient tensor of shape (B,).
    """

    drift = None
    diffusion = None
    ################################################################
    # TODO
    beta_t = beta_start + (beta_end - beta_start) * t
    drift = -0.5 * beta_t.view(-1, 1, 1, 1) * x
    diffusion = torch.sqrt(beta_t).view((-1, 1, 1, 1))
    ################################################################

    return drift, diffusion

def marginal_prob(t, beta_start=0.1, beta_end=20):

    """
    Computes the mean and standard deviation of the VP SDE marginal distribution.

    Args:
        t (Tensor): Continuous time tensor of shape (B,).
        beta_start (float): First beta value.
        beta_end (float): Last beta value.

    Returns:
        mean (Tensor): Mean tensor of shape (B,).
        std (Tensor): Standard deviation tensor of shape (B,).
    """

    mean = None
    std = None
    ################################################################
    # TODO
    beta_int = beta_start * t + 0.5 * (beta_end - beta_start) * t ** 2
    mean = torch.exp(-0.5 * beta_int)
    std = torch.sqrt(1.0 - torch.exp(-beta_int))
    ################################################################

    return mean, std

def q_sample(x0, t, noise, beta_start=0.1, beta_end=20):

    """
    Samples x_t from q(x_t | x_0) for the VP SDE.

    Args:
        x0 (Tensor): Clean image batch of shape (B, C, H, W).
        t (Tensor): Continuous time tensor of shape (B,).
        noise (Tensor): Gaussian noise with the same shape as x0.
        beta_start (float): First beta value.
        beta_end (float): Last beta value.

    Returns:
        xt (Tensor): Noised image batch of shape (B, C, H, W).
    """

    xt = None
    ################################################################
    # TODO
    mean, std = marginal_prob(t, beta_start=beta_start, beta_end=beta_end)
    mean = mean.view(-1, 1, 1, 1)
    std = std.view(-1, 1, 1, 1)

    xt = mean * x0 + std * noise
    ################################################################
    return xt

@torch.inference_mode()
def vp_sde_sample(model, num_samples, image_size, num_steps, eps, device, beta_start=0.1, beta_end=20):

    """
    Generates images with Euler-Maruyama sampling for the VP reverse SDE.

    Args:
        model (nn.Module): Trained score network.
        num_samples (int): Number of samples to generate.
        image_size (int): Spatial image size.
        num_steps (int): Number of discretization steps.
        eps (float): End time for sampling.
        device (torch.device): Device used for computation.
        beta_start (float): First beta value.
        beta_end (float): Last beta value.

    Returns:
        samples (Tensor): Generated image batch of shape (num_samples, 3, image_size, image_size).
        samples_trajectories (list): List of intermediate samples for visualization.
    """

    samples, samples_trajectories = None, []
    ################################################################
    # TODO
    was_training = model.training
    model.eval()

    xt = torch.randn(num_samples, model.in_channels, image_size, image_size, device=device)
    samples_trajectories = [xt.cpu()]

    timesteps = torch.linspace(1.0, eps, steps=num_steps, device=device)
    delta_t = timesteps[0] - timesteps[1]
    for t_index in timesteps:
        t = torch.ones(num_samples, device=device) * t_index
        drift, diffusion = vp_coeffs(xt, t, beta_start=beta_start, beta_end=beta_end)
        xt = xt - (drift - diffusion ** 2 * model(xt, t * 999)) * delta_t + diffusion * math.sqrt(delta_t) * torch.randn_like(xt)
        samples_trajectories.append(xt.cpu())
    samples = xt

    model.train(was_training)
    ################################################################
    
    return samples, samples_trajectories