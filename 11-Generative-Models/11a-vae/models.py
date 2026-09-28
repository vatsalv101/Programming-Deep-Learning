import torch
import torch.nn as nn

import utils


class VAE(nn.Module):
    """
    Convolutional variational autoencoder.

    Args:
        latent_dim (int): Dimensionality of the latent representation.
        device (torch.device | str | None): Device for the model parameters.
    """

    def __init__(self, latent_dim=128, device=None):
        super().__init__()
        ################################################################
        # TODO
        self.latent_dim = latent_dim
        self.device = torch.device("cpu") if device is None else torch.device(device)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.LeakyReLU(),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Flatten(),
        )

        self.fc_mu = nn.Linear(128 *4 *4, latent_dim)
        self.fc_logvar = nn.Linear(128 *4 *4, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 128*4*4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh(),
        )

        self.to(self.device)
        ################################################################


    def encode(self, x):
        """
        Encodes an image batch into latent distribution parameters.

        Args:
            x (Tensor): Image batch of shape (B, 3, 32, 32).

        Returns:
            mu (Tensor): Mean of the latent distribution of shape (B, latent_dim).
            logvar (Tensor): Log-variance of the latent distribution of shape (B, latent_dim).
        """
        mu, logvar = None, None
        ################################################################
        # TODO
        
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        ################################################################
        return mu, logvar


    def reparameterize(self, mu, logvar):
        """
        Samples latent vectors using the reparameterization trick.

        Args:
            mu (Tensor): Mean of the latent distribution of shape (B, latent_dim).
            logvar (Tensor): Log-variance of the latent distribution of shape (B, latent_dim).

        Returns:
            z (Tensor): Sampled latent vectors of shape (B, latent_dim).
        """
        z = None
        ################################################################
        # TODO
        
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        ################################################################
        return z


    def decode(self, z):
        """
        Decodes latent vectors into images.

        Args:
            z (Tensor): Latent vectors of shape (B, latent_dim).

        Returns:
            recon (Tensor): Reconstructed image batch of shape (B, 3, 32, 32).
        """
        recon = None
        ################################################################
        # TODO
        
        h = self.decoder_input(z)
        h = h.view(-1, 128, 4, 4)
        recon = self.decoder(h)
        ################################################################
        return recon


    def forward(self, x):
        """
        Performs a full VAE forward pass.

        Args:
            x (Tensor): Image batch of shape (B, 3, 32, 32).

        Returns:
            recon (Tensor): Reconstructed image batch of shape (B, 3, 32, 32).
            mu (Tensor): Mean of the latent distribution of shape (B, latent_dim).
            logvar (Tensor): Log-variance of the latent distribution of shape (B, latent_dim).
        """
        recon, mu, logvar = None, None, None
        ################################################################
        # TODO
        
        mu, logvar = self.encode(x)
        recon = self.decode(self.reparameterize(mu, logvar))

        ################################################################
        return recon, mu, logvar
    

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

        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decode(z)

        self.train(was_training)

        ################################################################
        return samples
    

    @torch.inference_mode()
    def interpolate(self, steps, device, x1=None, x2=None, method="linear"):
        """
        Generates images by interpolating between two latent vectors.

        If input images are provided, their latent representations are used as
        interpolation endpoints. Otherwise, random latent vectors are sampled.

        Args:
            steps (int): Number of interpolation steps.
            device (torch.device): Device used for computation.
            x1 (Tensor, optional): First image of shape (3, 32, 32).
            x2 (Tensor, optional): Second image of shape (3, 32, 32).
            method (str): Interpolation method, either "linear" or "slerp".

        Returns:
            samples (Tensor): Decoded interpolation images of shape (steps, 3, 32, 32).
        """
        samples = None
        ################################################################
        # TODO
        was_training = self.training
        self.eval()
        if x1 is not None:
            mu1, logvar1 = self.encode(x1.unsqueeze(0))
            z1 = self.reparameterize(mu1, logvar1)
        else:
            z1 = torch.randn(1, self.fc_mu.out_features).to(device)

        if x2 is not None:
            mu2, logvar2 = self.encode(x2.unsqueeze(0))
            z2 = self.reparameterize(mu2, logvar2)
        else:
            z2 = torch.randn(1, self.fc_mu.out_features).to(device)

        if method == "linear":
            z_interp = utils.linear_interpolation(z1.squeeze(0), z2.squeeze(0), steps)
        elif method == "slerp":
            z_interp = utils.slerp_interpolation(z1.squeeze(0), z2.squeeze(0), steps)
        else:
            raise ValueError(f"Invalid interpolation method: {method}")
        
        samples = self.decode(z_interp)

        self.train(was_training)
        ################################################################
        return samples