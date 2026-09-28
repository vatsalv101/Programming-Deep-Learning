import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import trainer

def _count_parameters(module):
    """
    Counts all learnable parameters inside a module.
    """
    return sum(parameter.numel() for parameter in module.parameters())

def _expected_gan_parameter_counts(latent_dim):
    """
    Returns expected parameter counts for the prescribed GAN architecture.
    """
    hidden_dim = 128 * 4 * 4

    generator_proj = latent_dim * hidden_dim + hidden_dim

    generator = (
        (128 * 64 * 4 * 4 + 64) + 2 * 64
        + (64 * 32 * 4 * 4 + 32) + 2 * 32
        + (32 * 3 * 4 * 4 + 3)
    )

    discriminator = (
        (3 * 32 * 4 * 4 + 32)
        + (32 * 64 * 4 * 4 + 64) + 2 * 64
        + (64 * 128 * 4 * 4 + 128) + 2 * 128
    )

    discriminator_proj = hidden_dim * 1 + 1

    return {
        "generator_proj": generator_proj,
        "generator": generator,
        "generator_total": generator_proj + generator,
        "discriminator": discriminator,
        "discriminator_proj": discriminator_proj,
        "discriminator_total": discriminator + discriminator_proj,
    }


def _contains_module(module, module_type):
    """
    Checks recursively whether a module contains a given module type.
    """
    return any(isinstance(submodule, module_type) for submodule in module.modules())


def test_gan_models():

    torch.manual_seed(0)

    batch_size = 4

    for latent_dim in [1, 17, 128]:
        try:
            gen = models.Generator(latent_dim=latent_dim)
        except Exception as error:
            print(f"❌ FAIL: Could not instantiate Generator with latent_dim={latent_dim} ({error}).")
            return False

        try:
            disc = models.Discriminator()
        except Exception as error:
            print(f"❌ FAIL: Could not instantiate Discriminator ({error}).")
            return False

        # Required attributes
        for attribute in ["latent_dim", "proj", "generator"]:
            if not hasattr(gen, attribute):
                print(f"❌ FAIL: Generator is missing required attribute `{attribute}`.")
                return False

        for attribute in ["discriminator", "proj"]:
            if not hasattr(disc, attribute):
                print(f"❌ FAIL: Discriminator is missing required attribute `{attribute}`.")
                return False

        if not isinstance(gen.proj, nn.Linear):
            print("❌ FAIL: Generator `proj` should be a Linear layer.")
            return False

        if not isinstance(disc.proj, nn.Sequential):
            print("❌ FAIL: Discriminator `proj` should be a Sequential containing Linear and Sigmoid.")
            return False

        if len(disc.proj) != 2:
            print("❌ FAIL: Discriminator `proj` should contain exactly Linear and Sigmoid.")
            return False

        if not isinstance(disc.proj[0], nn.Linear):
            print("❌ FAIL: Discriminator `proj[0]` should be a Linear layer.")
            return False

        if not isinstance(disc.proj[1], nn.Sigmoid):
            print("❌ FAIL: Discriminator `proj[1]` should be a Sigmoid layer.")
            return False

        if gen.proj.in_features != latent_dim:
            print("❌ FAIL: Generator `proj` should take latent_dim features.")
            return False

        if gen.proj.out_features != 128 * 4 * 4:
            print("❌ FAIL: Generator `proj` should output 128 * 4 * 4 features.")
            return False

        if disc.proj[0].in_features != 128 * 4 * 4:
            print("❌ FAIL: Discriminator Linear layer should take 128 * 4 * 4 features.")
            return False

        if disc.proj[0].out_features != 1:
            print("❌ FAIL: Discriminator Linear layer should output one value.")
            return False

        # Parameter counts
        expected = _expected_gan_parameter_counts(latent_dim)

        checks = [
            (gen.proj, expected["generator_proj"], "Generator proj"),
            (gen.generator, expected["generator"], "Generator conv stack"),
            (gen, expected["generator_total"], "Generator"),
            (disc.discriminator, expected["discriminator"], "Discriminator conv stack"),
            (disc.proj, expected["discriminator_proj"], "Discriminator proj"),
            (disc, expected["discriminator_total"], "Discriminator"),
        ]

        for module, expected_count, name in checks:
            actual_count = _count_parameters(module)
            if actual_count != expected_count:
                print(
                    f"❌ FAIL: Expected {name} with latent_dim={latent_dim} "
                    f"to have {expected_count} parameters, got {actual_count}."
                )
                return False

        # Generator forward
        z = torch.randn(batch_size, latent_dim, requires_grad=True)

        try:
            x_fake = gen(z)
        except Exception as error:
            print(f"❌ FAIL: `Generator(z)` raised an error ({error}).")
            return False

        if x_fake.shape != (batch_size, 3, 32, 32):
            print(f"❌ FAIL: Generator output should have shape ({batch_size}, 3, 32, 32).")
            return False

        if not torch.isfinite(x_fake).all():
            print("❌ FAIL: Generator output should contain only finite values.")
            return False

        if x_fake.min() < -1.0001 or x_fake.max() > 1.0001:
            print("❌ FAIL: Generator output should be in [-1, 1]. Did you forget final Tanh?")
            return False

        try:
            x_fake.mean().backward(retain_graph=True)
        except Exception as error:
            print(f"❌ FAIL: Backpropagation through Generator failed ({error}).")
            return False

        if z.grad is None:
            print("❌ FAIL: Generator should preserve gradients with respect to z.")
            return False

        # Discriminator forward
        x = torch.randn(batch_size, 3, 32, 32, requires_grad=True)

        try:
            probs = disc(x)
        except Exception as error:
            print(f"❌ FAIL: `Discriminator(x)` raised an error ({error}).")
            return False

        if probs.shape != (batch_size, 1):
            print(f"❌ FAIL: Discriminator output should have shape ({batch_size}, 1).")
            return False

        if not torch.isfinite(probs).all():
            print("❌ FAIL: Discriminator output should contain only finite values.")
            return False

        if probs.min() < -1e-6 or probs.max() > 1 + 1e-6:
            print("❌ FAIL: Discriminator output should be probabilities in [0, 1].")
            return False

        if not _contains_module(disc, nn.Sigmoid):
            print("❌ FAIL: Discriminator should end with a Sigmoid and output probabilities.")
            return False

        try:
            probs.mean().backward()
        except Exception as error:
            print(f"❌ FAIL: Backpropagation through Discriminator failed ({error}).")
            return False

        if x.grad is None:
            print("❌ FAIL: Discriminator should preserve gradients with respect to x.")
            return False

    print("✅ PASS: GAN Generator/Discriminator models look correct.")
    return True

def test_gan_criterion():

    torch.manual_seed(0)

    probs = torch.tensor(
        [[0.01], [0.12], [0.50], [0.88], [0.99]],
        requires_grad=True,
    )

    targets = torch.tensor(
        [[0.0], [1.0], [0.0], [1.0], [1.0]],
    )

    try:
        loss = trainer.gan_criterion(probs, targets)
    except Exception as error:
        print(f"❌ FAIL: `gan_criterion` raised an error ({error}).")
        return False

    expected = F.binary_cross_entropy(probs, targets)

    if loss.shape != ():
        print("❌ FAIL: `gan_criterion` should return a scalar tensor.")
        return False

    if not torch.isfinite(loss):
        print("❌ FAIL: `gan_criterion` should return a finite value.")
        return False

    if not torch.allclose(loss, expected, atol=1e-6, rtol=1e-6):
        print("❌ FAIL: `gan_criterion` does not match `F.binary_cross_entropy`.")
        return False

    try:
        loss.backward()
    except Exception as error:
        print(f"❌ FAIL: Backpropagation through `gan_criterion` failed ({error}).")
        return False

    if probs.grad is None:
        print("❌ FAIL: `probs` should receive gradients.")
        return False

    if not torch.isfinite(probs.grad).all():
        print("❌ FAIL: `probs.grad` should contain only finite values.")
        return False

    print("✅ PASS: GAN criterion is correct.")
    return True

def test_gan_step():

    torch.manual_seed(0)

    class _TinyGenerator(nn.Module):
        def __init__(self, latent_dim=3):
            super().__init__()
            self.latent_dim = latent_dim
            self.weight = nn.Parameter(torch.tensor(0.25))
            self.last_training_state = None
            self.last_grad_enabled = None
            self.last_z_device = None

        def forward(self, z):
            self.last_training_state = self.training
            self.last_grad_enabled = torch.is_grad_enabled()
            self.last_z_device = z.device

            batch_size = z.shape[0]
            value = self.weight * z.mean(dim=1).view(batch_size, 1, 1, 1)
            return value.expand(batch_size, 3, 8, 8)

    class _TinyDiscriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.5))
            self.last_training_state = None
            self.last_grad_enabled = None
            self.last_input_device = None

        def forward(self, x):
            self.last_training_state = self.training
            self.last_grad_enabled = torch.is_grad_enabled()
            self.last_input_device = x.device

            value = self.weight * x.mean(dim=(1, 2, 3)).view(-1, 1)
            return torch.sigmoid(value)

    def _check_metrics(metrics):
        expected_keys = {
            "gen_loss",
            "disc_loss",
            "disc_loss_real",
            "disc_loss_fake",
        }

        if set(metrics.keys()) != expected_keys:
            print("❌ FAIL: `_step` returned the wrong metric keys.")
            return False

        for key, value in metrics.items():
            if not isinstance(value, float):
                print(f"❌ FAIL: Metric `{key}` should be a Python float.")
                return False

            if not torch.isfinite(torch.tensor(value)):
                print(f"❌ FAIL: Metric `{key}` should be finite.")
                return False

            if value < 0:
                print(f"❌ FAIL: Metric `{key}` should be non-negative.")
                return False

        expected = 0.5 * (
            metrics["disc_loss_real"] + metrics["disc_loss_fake"]
        )

        if abs(metrics["disc_loss"] - expected) > 1e-6:
            print("❌ FAIL: Discriminator loss aggregation is incorrect.")
            return False

        return True

    def _check_recorded_calls(recorded_calls):
        if len(recorded_calls) != 3:
            print("❌ FAIL: `_step` used the criterion an unexpected number of times.")
            return False

        expected_targets = [
            torch.ones_like(recorded_calls[0][1]),
            torch.zeros_like(recorded_calls[1][1]),
            torch.ones_like(recorded_calls[2][1]),
        ]

        for idx, ((_, targets), expected) in enumerate(zip(recorded_calls, expected_targets)):
            if not torch.allclose(targets, expected):
                print(f"❌ FAIL: Criterion call {idx + 1} used incorrect targets.")
                return False

        if torch.allclose(recorded_calls[0][0], recorded_calls[1][0]):
            print("❌ FAIL: Criterion call 2 used incorrect discriminator outputs.")
            return False

        if torch.allclose(recorded_calls[0][0], recorded_calls[2][0]):
            print("❌ FAIL: Criterion call 3 used incorrect discriminator outputs.")
            return False

        return True

    device = torch.device("cpu")
    x = torch.ones(4, 3, 8, 8)

    original_gan_criterion = trainer.gan_criterion
    original_randn = trainer.torch.randn

    def fake_randn(*size, **kwargs):
        if len(size) == 1 and isinstance(size[0], tuple):
            size = size[0]

        return torch.ones(
            *size,
            device=kwargs.get("device", None),
            dtype=kwargs.get("dtype", torch.float32),
        )

    def run_case(is_train):
        gen = _TinyGenerator(latent_dim=3)
        disc = _TinyDiscriminator()

        gen_before = gen.weight.detach().clone()
        disc_before = disc.weight.detach().clone()

        if is_train:
            optimizers = (
                torch.optim.SGD(gen.parameters(), lr=0.1),
                torch.optim.SGD(disc.parameters(), lr=0.1),
            )
        else:
            optimizers = None

        recorded_calls = []

        def fake_gan_criterion(probs, targets):
            recorded_calls.append((
                probs.detach().clone(),
                targets.detach().clone(),
            ))
            return ((probs - targets) ** 2).mean()

        trainer.gan_criterion = fake_gan_criterion
        trainer.torch.randn = fake_randn

        try:
            metrics = trainer._step(
                gen,
                disc,
                x,
                device,
                optimizers=optimizers,
            )
        except Exception as error:
            print(f"❌ FAIL: `_step` raised an error ({error}).")
            return False
        finally:
            trainer.gan_criterion = original_gan_criterion
            trainer.torch.randn = original_randn

        if gen.last_training_state is not is_train:
            print("❌ FAIL: Generator mode was set incorrectly.")
            return False

        if disc.last_training_state is not is_train:
            print("❌ FAIL: Discriminator mode was set incorrectly.")
            return False

        if gen.last_grad_enabled is not is_train:
            print("❌ FAIL: Generator gradient mode was incorrect.")
            return False

        if disc.last_grad_enabled is not is_train:
            print("❌ FAIL: Discriminator gradient mode was incorrect.")
            return False

        if gen.last_z_device != device:
            print("❌ FAIL: Latent vectors were created on the wrong device.")
            return False

        if disc.last_input_device != device:
            print("❌ FAIL: Input batch was not moved to the selected device.")
            return False

        gen_changed = not torch.allclose(gen.weight.detach(), gen_before)
        disc_changed = not torch.allclose(disc.weight.detach(), disc_before)

        if is_train and not gen_changed:
            print("❌ FAIL: Generator parameters were not updated during training.")
            return False

        if is_train and not disc_changed:
            print("❌ FAIL: Discriminator parameters were not updated during training.")
            return False

        if not is_train and gen_changed:
            print("❌ FAIL: Generator parameters changed during evaluation.")
            return False

        if not is_train and disc_changed:
            print("❌ FAIL: Discriminator parameters changed during evaluation.")
            return False

        if not _check_metrics(metrics):
            return False

        if not _check_recorded_calls(recorded_calls):
            return False

        return True

    if not run_case(is_train=True):
        return False

    if not run_case(is_train=False):
        return False

    print("✅ PASS: GAN _step is correct.")
    return True