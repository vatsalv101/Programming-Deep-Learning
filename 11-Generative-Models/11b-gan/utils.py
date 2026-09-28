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
    
    

    
    ################################################################
    return interpolated