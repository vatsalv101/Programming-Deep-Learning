import torch
from torch import nn
import torch.nn.functional as F

import models
import trainer
import utils


def _count_parameters(module):
    """
    Counts all learnable parameters inside a module.
    """
    return sum(parameter.numel() for parameter in module.parameters())


def _expected_vae_parameter_counts(latent_dim):
    """
    Returns expected parameter counts for the prescribed VAE architecture.
    """
    hidden_dim = 128 * 4 * 4

    encoder = (
        (3 * 32 * 4 * 4 + 32)
        + (32 * 64 * 4 * 4 + 64) + 2 * 64
        + (64 * 128 * 4 * 4 + 128) + 2 * 128
    )

    fc_mu = hidden_dim * latent_dim + latent_dim
    fc_logvar = hidden_dim * latent_dim + latent_dim
    decoder_input = latent_dim * hidden_dim + hidden_dim

    decoder = (
        (128 * 64 * 4 * 4 + 64) + 2 * 64
        + (64 * 32 * 4 * 4 + 32) + 2 * 32
        + (32 * 3 * 4 * 4 + 3)
    )

    return {
        "encoder": encoder,
        "fc_mu": fc_mu,
        "fc_logvar": fc_logvar,
        "decoder_input": decoder_input,
        "decoder": decoder,
        "total": encoder + fc_mu + fc_logvar + decoder_input + decoder,
    }


def test_vae_architecture():
    """
    Tests whether the VAE defines the required components and has the exact
    number of parameters prescribed by the architecture.
    """
    for latent_dim in [1, 17, 128]:
        try:
            model = models.VAE(latent_dim=latent_dim)
        except Exception as error:
            print(f"❌ FAIL: Could not instantiate VAE with latent_dim={latent_dim} ({error}).")
            return False

        required_attributes = ["encoder", "fc_mu", "fc_logvar", "decoder_input", "decoder"]
        for attribute in required_attributes:
            if not hasattr(model, attribute):
                print(f"❌ FAIL: VAE is missing required attribute `{attribute}`.")
                return False

        if not isinstance(model.fc_mu, nn.Linear):
            print("❌ FAIL: `fc_mu` should be a Linear layer.")
            return False

        if not isinstance(model.fc_logvar, nn.Linear):
            print("❌ FAIL: `fc_logvar` should be a Linear layer.")
            return False

        if not isinstance(model.decoder_input, nn.Linear):
            print("❌ FAIL: `decoder_input` should be a Linear layer.")
            return False

        if model.fc_mu.out_features != latent_dim:
            print("❌ FAIL: `fc_mu` should output latent_dim features.")
            return False

        if model.fc_logvar.out_features != latent_dim:
            print("❌ FAIL: `fc_logvar` should output latent_dim features.")
            return False

        if model.decoder_input.in_features != latent_dim:
            print("❌ FAIL: `decoder_input` should take latent_dim features.")
            return False

        expected = _expected_vae_parameter_counts(latent_dim)

        checks = [
            (model.encoder, expected["encoder"], "encoder"),
            (model.fc_mu, expected["fc_mu"], "fc_mu"),
            (model.fc_logvar, expected["fc_logvar"], "fc_logvar"),
            (model.decoder_input, expected["decoder_input"], "decoder_input"),
            (model.decoder, expected["decoder"], "decoder"),
            (model, expected["total"], "VAE"),
        ]

        for module, expected_count, name in checks:
            actual_count = _count_parameters(module)
            if actual_count != expected_count:
                print(
                    f"❌ FAIL: Expected {name} with latent_dim={latent_dim} "
                    f"to have {expected_count} parameters, got {actual_count}."
                )
                return False

    print("✅ PASS: VAE architecture and parameter count look correct.")
    return True


def test_vae_modules():
    """
    Tests encode, decode, forward, and sample for correct shapes, finite values,
    and gradient behavior.
    """
    torch.manual_seed(0)

    latent_dim = 17
    batch_size = 4

    try:
        model = models.VAE(latent_dim=latent_dim)
    except Exception as error:
        print(f"❌ FAIL: Could not instantiate VAE ({error}).")
        return False

    x = torch.randn(batch_size, 3, 32, 32)

    try:
        mu, logvar = model.encode(x)
    except Exception as error:
        print(f"❌ FAIL: `encode(x)` raised an error ({error}).")
        return False

    if mu.shape != (batch_size, latent_dim):
        print(f"❌ FAIL: `mu` should have shape ({batch_size}, {latent_dim}).")
        return False

    if logvar.shape != (batch_size, latent_dim):
        print(f"❌ FAIL: `logvar` should have shape ({batch_size}, {latent_dim}).")
        return False

    z = torch.randn(batch_size, latent_dim, requires_grad=True)

    try:
        decoded = model.decode(z)
    except Exception as error:
        print(f"❌ FAIL: `decode(z)` raised an error ({error}).")
        return False

    if decoded.shape != (batch_size, 3, 32, 32):
        print(f"❌ FAIL: `decode(z)` should return shape ({batch_size}, 3, 32, 32).")
        return False

    try:
        decoded.mean().backward()
    except Exception as error:
        print(f"❌ FAIL: Backpropagation through `decode(z)` failed ({error}).")
        return False

    if z.grad is None:
        print("❌ FAIL: `decode(z)` should preserve gradients with respect to z.")
        return False

    try:
        output = model(x)
    except Exception as error:
        print(f"❌ FAIL: `model(x)` raised an error ({error}).")
        return False

    if not isinstance(output, tuple) or len(output) != 3:
        print("❌ FAIL: `model(x)` should return `(recon, mu, logvar)`.")
        return False

    recon, mu, logvar = output

    if recon.shape != (batch_size, 3, 32, 32):
        print(f"❌ FAIL: Forward reconstruction should have shape ({batch_size}, 3, 32, 32).")
        return False

    if mu.shape != (batch_size, latent_dim) or logvar.shape != (batch_size, latent_dim):
        print(f"❌ FAIL: Forward `mu` and `logvar` should have shape ({batch_size}, {latent_dim}).")
        return False

    if not torch.isfinite(recon).all() or not torch.isfinite(mu).all() or not torch.isfinite(logvar).all():
        print("❌ FAIL: Forward pass should return finite tensors.")
        return False

    try:
        samples = model.sample(num_samples=5, device=torch.device("cpu"))
    except Exception as error:
        print(f"❌ FAIL: `sample(num_samples, device)` raised an error ({error}).")
        return False

    if samples.shape != (5, 3, 32, 32):
        print("❌ FAIL: `sample(5, device)` should return shape (5, 3, 32, 32).")
        return False

    if samples.requires_grad:
        print("❌ FAIL: `sample` should run without gradient tracking.")
        return False

    if not torch.isfinite(samples).all():
        print("❌ FAIL: `sample` should return finite values.")
        return False

    print("✅ PASS: VAE encode/decode/forward/sample look correct.")
    return True


def test_vae_reparameterize():
    """
    Tests whether reparameterize implements z = mu + eps * exp(0.5 * logvar)
    and preserves gradients.
    """
    torch.manual_seed(0)

    latent_dim = 5
    model = models.VAE(latent_dim=latent_dim)

    mu = torch.randn(3, latent_dim, requires_grad=True)
    logvar = torch.randn(3, latent_dim, requires_grad=True)
    eps = torch.linspace(-1.0, 1.0, steps=mu.numel()).view_as(mu)

    original_randn_like = models.torch.randn_like

    def fake_randn_like(tensor):
        return eps.to(device=tensor.device, dtype=tensor.dtype)

    models.torch.randn_like = fake_randn_like

    try:
        z = model.reparameterize(mu, logvar)
    except Exception as error:
        print(f"❌ FAIL: `reparameterize(mu, logvar)` raised an error ({error}).")
        models.torch.randn_like = original_randn_like
        return False
    finally:
        models.torch.randn_like = original_randn_like

    expected = mu + eps * torch.exp(0.5 * logvar)

    if z.shape != mu.shape:
        print("❌ FAIL: `reparameterize` should return a tensor with the same shape as mu.")
        return False

    if not torch.allclose(z, expected, atol=1e-6, rtol=1e-6):
        print("❌ FAIL: `reparameterize` used the wrong formula.")
        return False

    try:
        z.sum().backward()
    except Exception as error:
        print(f"❌ FAIL: Backpropagation through `reparameterize` failed ({error}).")
        return False

    if mu.grad is None:
        print("❌ FAIL: `mu` should receive gradients through `reparameterize`.")
        return False

    if logvar.grad is None:
        print("❌ FAIL: `logvar` should receive gradients through `reparameterize`.")
        return False

    print("✅ PASS: VAE reparameterization is correct.")
    return True


def test_vae_loss():
    """
    Tests the VAE loss value, output format, and gradient flow.
    """
    torch.manual_seed(0)

    x = torch.randn(3, 2, 4, 4)
    recon = torch.randn(3, 2, 4, 4, requires_grad=True)
    mu = torch.randn(3, 5, requires_grad=True)
    logvar = torch.randn(3, 5, requires_grad=True)

    try:
        output = trainer.vae_loss(recon, x, mu, logvar)
    except Exception as error:
        print(f"❌ FAIL: `vae_loss` raised an error ({error}).")
        return False

    if not isinstance(output, tuple) or len(output) != 3:
        print("❌ FAIL: `vae_loss` should return `(loss, recon_loss, kl_loss)`.")
        return False

    loss, recon_loss, kl_loss = output

    for name, value in [("loss", loss), ("recon_loss", recon_loss), ("kl_loss", kl_loss)]:
        if value.shape != ():
            print(f"❌ FAIL: `{name}` should be a scalar tensor.")
            return False

        if not torch.isfinite(value):
            print(f"❌ FAIL: `{name}` should be finite.")
            return False

    batch_size = x.shape[0]
    expected_recon_loss = F.mse_loss(recon, x, reduction="sum") / batch_size
    expected_kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    expected_loss = expected_recon_loss + expected_kl_loss

    if not torch.allclose(recon_loss, expected_recon_loss, atol=1e-6, rtol=1e-6):
        print("❌ FAIL: Reconstruction loss has the wrong value.")
        return False

    if not torch.allclose(kl_loss, expected_kl_loss, atol=1e-6, rtol=1e-6):
        print("❌ FAIL: KL loss has the wrong value.")
        return False

    if not torch.allclose(loss, expected_loss, atol=1e-6, rtol=1e-6):
        print("❌ FAIL: Total loss should be `recon_loss + kl_loss`.")
        return False

    try:
        loss.backward()
    except Exception as error:
        print(f"❌ FAIL: Backpropagation through `vae_loss` failed ({error}).")
        return False

    if recon.grad is None:
        print("❌ FAIL: `recon` should receive gradients.")
        return False

    if mu.grad is None:
        print("❌ FAIL: `mu` should receive gradients.")
        return False

    if logvar.grad is None:
        print("❌ FAIL: `logvar` should receive gradients.")
        return False

    print("✅ PASS: VAE loss is correct.")
    return True


class _TinyVAE(nn.Module):
    """
    Minimal VAE-like model for testing _step.
    """
    def __init__(self, latent_dim=3):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.mu = nn.Parameter(torch.zeros(1, latent_dim))
        self.logvar = nn.Parameter(torch.zeros(1, latent_dim))

        self.last_training_state = None
        self.last_grad_enabled = None
        self.last_input_device = None

    def forward(self, x):
        self.last_training_state = self.training
        self.last_grad_enabled = torch.is_grad_enabled()
        self.last_input_device = x.device

        batch_size = x.shape[0]
        recon = self.scale * x
        mu = self.mu.expand(batch_size, -1)
        logvar = self.logvar.expand(batch_size, -1)

        return recon, mu, logvar


def test_vae_step():
    """
    Tests _step in training and evaluation mode.
    """
    torch.manual_seed(0)

    device = torch.device("cpu")
    x = torch.randn(4, 3, 8, 8)

    model = _TinyVAE()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    parameter_before = model.scale.detach().clone()

    try:
        train_metrics = trainer._step(model, x, device, optimizer)
    except Exception as error:
        print(f"❌ FAIL: `_step(..., optimizer)` raised an error ({error}).")
        return False

    if model.last_training_state is not True:
        print("❌ FAIL: `_step` should set the model to training mode when an optimizer is given.")
        return False

    if model.last_grad_enabled is not True:
        print("❌ FAIL: `_step` should enable gradients during training.")
        return False

    if model.last_input_device != device:
        print("❌ FAIL: `_step` should move the input batch to the selected device.")
        return False

    if torch.allclose(model.scale.detach(), parameter_before):
        print("❌ FAIL: `_step` should update model parameters during training.")
        return False

    expected_keys = {"loss", "recon_loss", "kl_loss"}

    if set(train_metrics.keys()) != expected_keys:
        print("❌ FAIL: `_step` should return loss, recon_loss, and kl_loss.")
        return False

    for key, value in train_metrics.items():
        if not isinstance(value, float):
            print(f"❌ FAIL: Metric `{key}` should be returned as a Python float.")
            return False

        if not torch.isfinite(torch.tensor(value)):
            print(f"❌ FAIL: Metric `{key}` should be finite.")
            return False

    model = _TinyVAE()
    parameter_before = model.scale.detach().clone()

    try:
        eval_metrics = trainer._step(model, x, device, optimizer=None)
    except Exception as error:
        print(f"❌ FAIL: `_step(..., optimizer=None)` raised an error ({error}).")
        return False

    if model.last_training_state is not False:
        print("❌ FAIL: `_step` should set the model to evaluation mode when no optimizer is given.")
        return False

    if model.last_grad_enabled is not False:
        print("❌ FAIL: `_step` should disable gradients during evaluation.")
        return False

    if not torch.allclose(model.scale.detach(), parameter_before):
        print("❌ FAIL: `_step` should not update parameters during evaluation.")
        return False

    if set(eval_metrics.keys()) != expected_keys:
        print("❌ FAIL: Evaluation metrics should contain loss, recon_loss, and kl_loss.")
        return False

    for key, value in eval_metrics.items():
        if not isinstance(value, float):
            print(f"❌ FAIL: Evaluation metric `{key}` should be returned as a Python float.")
            return False

        if not torch.isfinite(torch.tensor(value)):
            print(f"❌ FAIL: Evaluation metric `{key}` should be finite.")
            return False

    print("✅ PASS: VAE _step is correct.")
    return True


def test_latent_interpolation_utils():
    """
    Tests linear interpolation and SLERP for correct values, shapes, and endpoints.
    """
    z1 = torch.tensor([0.0, 2.0])
    z2 = torch.tensor([4.0, 6.0])

    try:
        interpolated = utils.linear_interpolation(z1, z2, steps=5)
    except Exception as error:
        print(f"❌ FAIL: `linear_interpolation` raised an error ({error}).")
        return False

    expected = torch.tensor([
        [0.0, 2.0],
        [1.0, 3.0],
        [2.0, 4.0],
        [3.0, 5.0],
        [4.0, 6.0],
    ])

    if interpolated.shape != (5, 2):
        print("❌ FAIL: `linear_interpolation` should return shape (steps, latent_dim).")
        return False

    if not torch.allclose(interpolated, expected, atol=1e-6):
        print("❌ FAIL: `linear_interpolation` returned wrong values.")
        return False

    z1 = torch.tensor([1.0, 0.0])
    z2 = torch.tensor([0.0, 1.0])

    try:
        interpolated = utils.slerp_interpolation(z1, z2, steps=3)
    except Exception as error:
        print(f"❌ FAIL: `slerp_interpolation` raised an error ({error}).")
        return False

    expected = torch.tensor([
        [1.0, 0.0],
        [2.0 ** -0.5, 2.0 ** -0.5],
        [0.0, 1.0],
    ])

    if interpolated.shape != (3, 2):
        print("❌ FAIL: `slerp_interpolation` should return shape (steps, latent_dim).")
        return False

    if not torch.allclose(interpolated, expected, atol=1e-6):
        print("❌ FAIL: `slerp_interpolation` returned wrong values for orthogonal vectors.")
        return False

    z1 = torch.tensor([2.0, 0.0])
    z2 = torch.tensor([4.0, 0.0])

    try:
        interpolated = utils.slerp_interpolation(z1, z2, steps=4)
    except Exception as error:
        print(f"❌ FAIL: SLERP fallback raised an error ({error}).")
        return False

    expected = utils.linear_interpolation(z1, z2, steps=4)

    if not torch.allclose(interpolated, expected, atol=1e-6):
        print("❌ FAIL: SLERP should fall back to linear interpolation for parallel vectors.")
        return False

    print("✅ PASS: Latent interpolation utilities are correct.")
    return True


def test_vae_interpolate():
    """
    Tests whether VAE.interpolate uses the selected interpolation method and
    returns decoded images with the correct shape.
    """
    torch.manual_seed(0)

    latent_dim = 7
    steps = 4
    device = torch.device("cpu")

    try:
        model = models.VAE(latent_dim=latent_dim).to(device)
    except Exception as error:
        print(f"❌ FAIL: Could not instantiate VAE ({error}).")
        return False

    target_utils = models.utils if hasattr(models, "utils") else utils

    original_linear = target_utils.linear_interpolation
    original_slerp = target_utils.slerp_interpolation

    calls = {"linear": 0, "slerp": 0}

    def fake_linear(z1, z2, num_steps):
        calls["linear"] += 1
        if z1.shape != (latent_dim,) or z2.shape != (latent_dim,):
            print("❌ FAIL: `interpolate` should pass latent vectors of shape (latent_dim,).")
        return torch.zeros(num_steps, latent_dim, device=z1.device)

    def fake_slerp(z1, z2, num_steps):
        calls["slerp"] += 1
        if z1.shape != (latent_dim,) or z2.shape != (latent_dim,):
            print("❌ FAIL: `interpolate` should pass latent vectors of shape (latent_dim,).")
        return torch.ones(num_steps, latent_dim, device=z1.device)

    target_utils.linear_interpolation = fake_linear
    target_utils.slerp_interpolation = fake_slerp

    try:
        linear_samples = model.interpolate(steps=steps, device=device, method="linear")
        slerp_samples = model.interpolate(steps=steps, device=device, method="slerp")
    except Exception as error:
        print(f"❌ FAIL: `interpolate` raised an error ({error}).")
        target_utils.linear_interpolation = original_linear
        target_utils.slerp_interpolation = original_slerp
        return False
    finally:
        target_utils.linear_interpolation = original_linear
        target_utils.slerp_interpolation = original_slerp

    if calls["linear"] != 1:
        print("❌ FAIL: `interpolate(..., method='linear')` should call `linear_interpolation`.")
        return False

    if calls["slerp"] != 1:
        print("❌ FAIL: `interpolate(..., method='slerp')` should call `slerp_interpolation`.")
        return False

    if linear_samples.shape != (steps, 3, 32, 32):
        print("❌ FAIL: Linear interpolation samples should have shape (steps, 3, 32, 32).")
        return False

    if slerp_samples.shape != (steps, 3, 32, 32):
        print("❌ FAIL: SLERP interpolation samples should have shape (steps, 3, 32, 32).")
        return False

    if linear_samples.requires_grad or slerp_samples.requires_grad:
        print("❌ FAIL: `interpolate` should run without gradient tracking.")
        return False

    if not torch.isfinite(linear_samples).all() or not torch.isfinite(slerp_samples).all():
        print("❌ FAIL: `interpolate` should return finite values.")
        return False

    print("✅ PASS: VAE interpolation is correct.")
    return True