import torch
import torch.nn as nn

import utils


class Generator(nn.Module):
    """
    Convolutional generator for GAN.

    Args:
        latent_dim (int): Dimensionality of the latent representation.
    """

    def __init__(self, latent_dim=128):
        super().__init__()
        ################################################################
        # TODO
        self.latent_dim = latent_dim
        self.proj = nn.Linear(self.latent_dim, 128 * 4 * 4)
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
        ################################################################

    def forward(self, z):
        """
        Performs a generator forward pass.

        Args:
            z (Tensor): Latent vectors of shape (B, latent_dim).

        Returns:
            x_fake (Tensor): Reconstructed image batch of shape (B, 3, 32, 32).
        """
        x_fake = None
        ################################################################
        # TODO
        h = self.proj(z)
        h = h.view(-1, 128, 4, 4)
        x_fake = self.generator(h)
        ################################################################
        return x_fake

    @torch.inference_mode()
    def sample(self, num_samples, device):
        """
        Samples new images from the latent prior.

        Args:
            num_samples (int): Number of images to generate.
            device (torch.device): Device used for computation.

        Returns:
            samples (Tensor): Generated image batch of shape (num_samples, 3, 32, 32).
        """
        samples = None
        ################################################################
        # TODO
        was_training = self.training
        self.eval()

        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self(z)

        self.train(was_training)
        ################################################################
        return samples

    @torch.inference_mode()
    def interpolate(self, steps, device, z1=None, z2=None, method="linear"):
        """
        Generates images by interpolating between two latent vectors.

        Args:
            steps (int): Number of interpolation steps.
            device (torch.device): Device used for computation.
            method (str): Interpolation method, either "linear" or "slerp".

        Returns:
            samples (Tensor): Decoded interpolation images of shape (steps, 3, 32, 32).
        """
        samples = None
        ################################################################
        # TODO
        was_training = self.training
        self.eval()

        if z1 is None:
            z1 = torch.randn(1, self.latent_dim).to(device)
        if z2 is None:
            z2 = torch.randn(1, self.latent_dim).to(device)

        if method == "linear":
            z_interp = utils.linear_interpolation(z1.squeeze(0), z2.squeeze(0), steps)
        elif method == "slerp":
            z_interp = utils.slerp_interpolation(z1.squeeze(0), z2.squeeze(0), steps)
        else:
            raise ValueError(f"Invalid interpolation method: {method}")

        samples = self(z_interp)

        self.train(was_training)
        ################################################################
        return samples


class Discriminator(nn.Module):
    """
    Convolutional discriminator for GAN.
    """

    def __init__(self):
        super().__init__()
        ################################################################
        # TODO
        self.discriminator = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1),
            nn.Sigmoid(),
        )
        ################################################################

    def forward(self, x):
        """
        Performs a discriminator forward pass.

        Args:
            x (Tensor): Input images of shape (B, 3, 32, 32).

        Returns:
            probs (Tensor): Discriminator probabilities of shape (B, 1).
        """
        probs = None
        ################################################################
        # TODO
        h = self.discriminator(x)
        probs = self.proj(h)
        ################################################################
        return probs