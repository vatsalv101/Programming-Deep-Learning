import torch
import torch.nn.functional as F


def linear_interpolation(z1, z2, steps):
    """
    Linearly interpolates between two latent vectors.

    Args:
        z1 (Tensor): Starting latent vector of shape (latent_dim,).
        z2 (Tensor): Ending latent vector of shape (latent_dim,).
        steps (int): Number of interpolation steps.

    Returns:
        interpolated (Tensor): Interpolated latent vectors of shape (steps, latent_dim).
    """
    interpolated = None
    ################################################################
    # TODO
    t = torch.linspace(0.0, 1.0, steps, device=z1.device, dtype=z1.dtype)
    interpolated = z1.unsqueeze(0) + (z2 - z1).unsqueeze(0) * t.unsqueeze(1)
    ################################################################
    return interpolated


def slerp_interpolation(z1, z2, steps):
    """
    Spherically interpolates between two latent vectors.

    The interpolation follows the shortest path on the hypersphere spanned by
    the two vectors. If the vectors are almost parallel, linear interpolation
    is used as a stable fallback.

    Args:
        z1 (Tensor): Starting latent vector of shape (latent_dim,).
        z2 (Tensor): Ending latent vector of shape (latent_dim,).
        steps (int): Number of interpolation steps.

    Returns:
        interpolated (Tensor): Interpolated latent vectors of shape (steps, latent_dim).
    """
    interpolated = None
    ################################################################
    # TODO
    z1 = z1.to(dtype=torch.float32)
    z2 = z2.to(dtype=torch.float32)

    v1_norm = torch.linalg.norm(z1)
    v2_norm = torch.linalg.norm(z2)
    zero = torch.tensor(0.0, device=z1.device, dtype=z1.dtype)
    if torch.isclose(v1_norm, zero) or torch.isclose(v2_norm, zero):
        return linear_interpolation(z1, z2, steps)

    dot = torch.dot(z1, z2)
    cos_theta = dot / (v1_norm * v2_norm)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)

    if torch.isclose(theta, zero) or torch.isclose(theta, torch.tensor(torch.pi, device=z1.device, dtype=z1.dtype)):
        return linear_interpolation(z1, z2, steps)

    t = torch.linspace(0.0, 1.0, steps, device=z1.device, dtype=z1.dtype)
    interpolated = (
        torch.sin((1 - t) * theta) / torch.sin(theta) * z1
        + torch.sin(t * theta) / torch.sin(theta) * z2
    )
    ################################################################
    return interpolated