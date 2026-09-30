import torch
from torch import nn

import utils

def test_vp_coeffs():
    torch.manual_seed(0)

    x = torch.tensor(
        [
            [[[0.0, 1.0],
              [2.0, 3.0]]],
            [[[4.0, 5.0],
              [6.0, 7.0]]],
            [[[8.0, 9.0],
              [10.0, 11.0]]],
        ]
    )

    t = torch.tensor([0.0, 0.5, 1.0])

    drift, diffusion = utils.vp_coeffs(
        x=x,
        t=t,
        beta_start=0.1,
        beta_end=0.5,
    )

    if not isinstance(drift, torch.Tensor):
        print("❌ FAIL: vp_coeffs should return drift as a tensor.")
        return False

    if not isinstance(diffusion, torch.Tensor):
        print("❌ FAIL: vp_coeffs should return diffusion as a tensor.")
        return False

    if drift.shape != x.shape:
        print("❌ FAIL: Drift should have the same shape as x.")
        return False

    if diffusion.shape != (3, 1, 1, 1):
        print("❌ FAIL: Diffusion should have shape (B, 1, 1, 1) for broadcasting.")
        return False

    if drift.dtype != x.dtype:
        print("❌ FAIL: Drift should have the same dtype as x.")
        return False

    if diffusion.dtype != x.dtype:
        print("❌ FAIL: Diffusion should have the same dtype as x.")
        return False

    if drift.device != x.device:
        print("❌ FAIL: Drift should be on the same device as x.")
        return False

    if diffusion.device != x.device:
        print("❌ FAIL: Diffusion should be on the same device as x.")
        return False

    if not torch.isfinite(drift).all():
        print("❌ FAIL: Drift should contain only finite values.")
        return False

    if not torch.isfinite(diffusion).all():
        print("❌ FAIL: Diffusion should contain only finite values.")
        return False

    expected_beta_t = torch.tensor([0.1, 0.3, 0.5])
    expected_drift = -0.5 * expected_beta_t.view(-1, 1, 1, 1) * x
    expected_diffusion = torch.tensor([0.31622776, 0.54772258, 0.70710677]).view(-1, 1, 1, 1)

    if not torch.allclose(drift, expected_drift, atol=1e-6):
        max_error = (drift - expected_drift).abs().max().item()
        print(f"❌ FAIL: Drift values are incorrect. Max error: {max_error:.2e}")
        return False

    if not torch.allclose(diffusion, expected_diffusion, atol=1e-6):
        max_error = (diffusion - expected_diffusion).abs().max().item()
        print(f"❌ FAIL: Diffusion values are incorrect. Max error: {max_error:.2e}")
        return False

    if not torch.all(diffusion[1:] > diffusion[:-1]):
        print("❌ FAIL: Diffusion values should increase over time.")
        return False

    x_train = x.clone().requires_grad_(True)

    drift_train, diffusion_train = utils.vp_coeffs(
        x=x_train,
        t=t,
        beta_start=0.1,
        beta_end=0.5,
    )

    loss = drift_train.square().mean() + diffusion_train.square().mean()
    loss.backward()

    if x_train.grad is None:
        print("❌ FAIL: vp_coeffs should support backpropagation through x.")
        return False

    if not torch.isfinite(x_train.grad).all():
        print("❌ FAIL: vp_coeffs gradients should contain only finite values.")
        return False

    if torch.cuda.is_available():
        device = torch.device("cuda")

        x_cuda = x.to(device)
        t_cuda = t.to(device)

        drift_cuda, diffusion_cuda = utils.vp_coeffs(
            x=x_cuda,
            t=t_cuda,
            beta_start=0.1,
            beta_end=0.5,
        )

        if drift_cuda.device.type != "cuda":
            print("❌ FAIL: Drift is on an unexpected device.")
            return False

        if diffusion_cuda.device.type != "cuda":
            print("❌ FAIL: Diffusion is on an unexpected device.")
            return False

    print("✅ PASS: vp_coeffs seems to be correct.")
    return True


def test_marginal_prob():
    torch.manual_seed(0)

    t = torch.tensor([0.0, 0.5, 1.0])

    mean, std = utils.marginal_prob(
        t=t,
        beta_start=0.1,
        beta_end=0.5,
    )

    if not isinstance(mean, torch.Tensor):
        print("❌ FAIL: marginal_prob should return mean as a tensor.")
        return False

    if not isinstance(std, torch.Tensor):
        print("❌ FAIL: marginal_prob should return std as a tensor.")
        return False

    if mean.shape != (3,):
        print("❌ FAIL: Mean should have shape (B,).")
        return False

    if std.shape != (3,):
        print("❌ FAIL: Standard deviation should have shape (B,).")
        return False

    if mean.dtype != t.dtype:
        print("❌ FAIL: Mean should have the same dtype as t.")
        return False

    if std.dtype != t.dtype:
        print("❌ FAIL: Standard deviation should have the same dtype as t.")
        return False

    if mean.device != t.device:
        print("❌ FAIL: Mean should be on the same device as t.")
        return False

    if std.device != t.device:
        print("❌ FAIL: Standard deviation should be on the same device as t.")
        return False

    if not torch.isfinite(mean).all():
        print("❌ FAIL: Mean should contain only finite values.")
        return False

    if not torch.isfinite(std).all():
        print("❌ FAIL: Standard deviation should contain only finite values.")
        return False

    expected_mean = torch.tensor([1.00000000, 0.95122945, 0.86070800])
    expected_std = torch.tensor([0.00000000, 0.30848432, 0.50909901])

    if not torch.allclose(mean, expected_mean, atol=1e-6):
        max_error = (mean - expected_mean).abs().max().item()
        print(f"❌ FAIL: Mean values are incorrect. Max error: {max_error:.2e}")
        return False

    if not torch.allclose(std, expected_std, atol=1e-6):
        max_error = (std - expected_std).abs().max().item()
        print(f"❌ FAIL: Standard deviation values are incorrect. Max error: {max_error:.2e}")
        return False

    if not torch.all(mean <= 1.0):
        print("❌ FAIL: Mean values should not exceed 1.")
        return False

    if not torch.all(mean > 0.0):
        print("❌ FAIL: Mean values should be positive.")
        return False

    if not torch.all(std >= 0.0):
        print("❌ FAIL: Standard deviation values should be non-negative.")
        return False

    if not torch.all(mean[1:] < mean[:-1]):
        print("❌ FAIL: Mean should decrease over time.")
        return False

    if not torch.all(std[1:] > std[:-1]):
        print("❌ FAIL: Standard deviation should increase over time.")
        return False

    consistency = mean ** 2 + std ** 2

    if not torch.allclose(consistency, torch.ones_like(consistency), atol=1e-6):
        print("❌ FAIL: marginal_prob outputs are internally inconsistent.")
        return False

    t_train = torch.tensor([0.1, 0.5, 1.0], requires_grad=True)

    mean_train, std_train = utils.marginal_prob(
        t=t_train,
        beta_start=0.1,
        beta_end=0.5,
    )

    loss = mean_train.mean() + std_train.mean()
    loss.backward()

    if t_train.grad is None:
        print("❌ FAIL: marginal_prob should support backpropagation through t.")
        return False

    if not torch.isfinite(t_train.grad).all():
        print("❌ FAIL: marginal_prob gradients should contain only finite values away from t = 0.")
        return False

    if torch.cuda.is_available():
        device = torch.device("cuda")

        t_cuda = t.to(device)

        mean_cuda, std_cuda = utils.marginal_prob(
            t=t_cuda,
            beta_start=0.1,
            beta_end=0.5,
        )

        if mean_cuda.device.type != "cuda":
            print("❌ FAIL: Mean is on an unexpected device.")
            return False

        if std_cuda.device.type != "cuda":
            print("❌ FAIL: Standard deviation is on an unexpected device.")
            return False

    print("✅ PASS: marginal_prob seems to be correct.")
    return True


def test_vp_q_sample():
    torch.manual_seed(0)

    x0 = torch.tensor(
        [
            [[[0.0, 1.0],
              [2.0, 3.0]]],
            [[[4.0, 5.0],
              [6.0, 7.0]]],
            [[[8.0, 9.0],
              [10.0, 11.0]]],
        ]
    )

    noise = torch.tensor(
        [
            [[[10.0, 11.0],
              [12.0, 13.0]]],
            [[[20.0, 21.0],
              [22.0, 23.0]]],
            [[[30.0, 31.0],
              [32.0, 33.0]]],
        ]
    )

    t = torch.tensor([0.0, 0.5, 1.0])

    xt = utils.q_sample(
        x0=x0,
        t=t,
        noise=noise,
        beta_start=0.1,
        beta_end=0.5,
    )

    expected = torch.tensor(
        [
            [[[0.0000000, 1.0000000],
              [2.0000000, 3.0000000]]],
            [[[9.9746037, 11.2343178],
              [12.4940319, 13.7537460]]],
            [[[22.1586342, 23.5284424],
              [24.8982487, 26.2680550]]],
        ]
    )

    if not isinstance(xt, torch.Tensor):
        print("❌ FAIL: q_sample should return a tensor.")
        return False

    if xt.shape != x0.shape:
        print("❌ FAIL: q_sample output should have the same shape as x0.")
        return False

    if xt.dtype != x0.dtype:
        print("❌ FAIL: q_sample output should have the same dtype as x0.")
        return False

    if xt.device != x0.device:
        print("❌ FAIL: q_sample output should be on the same device as x0.")
        return False

    if not torch.isfinite(xt).all():
        print("❌ FAIL: q_sample output should contain only finite values.")
        return False

    if not torch.allclose(xt, expected, atol=1e-6):
        max_error = (xt - expected).abs().max().item()
        print(f"❌ FAIL: q_sample values are incorrect. Max error: {max_error:.2e}")
        return False

    # Check that different batch elements can use different continuous times.
    x0_batch = torch.ones(3, 1, 2, 2)
    noise_batch = torch.ones_like(x0_batch) * 10.0
    t_batch = torch.tensor([0.0, 0.5, 1.0])

    xt_batch = utils.q_sample(
        x0=x0_batch,
        t=t_batch,
        noise=noise_batch,
        beta_start=0.1,
        beta_end=0.5,
    )

    expected_batch = torch.tensor([1.0000000, 4.0360727, 5.9516983]).view(3, 1, 1, 1).expand_as(x0_batch)

    if not torch.allclose(xt_batch, expected_batch, atol=1e-6):
        max_error = (xt_batch - expected_batch).abs().max().item()
        print(f"❌ FAIL: q_sample does not handle per-sample continuous times correctly. Max error: {max_error:.2e}")
        return False

    # Clean-image limit at t = 0.
    xt_clean = utils.q_sample(
        x0=x0,
        t=torch.zeros(3),
        noise=noise,
        beta_start=0.1,
        beta_end=0.5,
    )

    if not torch.allclose(xt_clean, x0, atol=1e-6):
        print("❌ FAIL: q_sample should preserve x0 at t = 0.")
        return False

    # Zero-noise check.
    xt_zero_noise = utils.q_sample(
        x0=x0,
        t=t,
        noise=torch.zeros_like(noise),
        beta_start=0.1,
        beta_end=0.5,
    )

    expected_mean = torch.tensor([1.00000000, 0.95122945, 0.86070800]).view(3, 1, 1, 1)
    expected_zero_noise = expected_mean * x0

    if not torch.allclose(xt_zero_noise, expected_zero_noise, atol=1e-6):
        max_error = (xt_zero_noise - expected_zero_noise).abs().max().item()
        print(f"❌ FAIL: q_sample does not handle zero noise correctly. Max error: {max_error:.2e}")
        return False

    xt_again = utils.q_sample(
        x0=x0,
        t=t,
        noise=noise,
        beta_start=0.1,
        beta_end=0.5,
    )

    if not torch.allclose(xt, xt_again, atol=1e-6):
        print("❌ FAIL: q_sample should be deterministic for fixed inputs.")
        return False

    x0_train = x0.clone().requires_grad_(True)

    xt_train = utils.q_sample(
        x0=x0_train,
        t=t,
        noise=noise,
        beta_start=0.1,
        beta_end=0.5,
    )

    loss = xt_train.square().mean()
    loss.backward()

    if x0_train.grad is None:
        print("❌ FAIL: q_sample should support backpropagation through x0.")
        return False

    if not torch.isfinite(x0_train.grad).all():
        print("❌ FAIL: q_sample gradients should contain only finite values.")
        return False

    if torch.cuda.is_available():
        device = torch.device("cuda")

        xt_cuda = utils.q_sample(
            x0=x0.to(device),
            t=t.to(device),
            noise=noise.to(device),
            beta_start=0.1,
            beta_end=0.5,
        )

        if xt_cuda.device.type != "cuda":
            print("❌ FAIL: q_sample output is on an unexpected device.")
            return False

    print("✅ PASS: VP q_sample seems to be correct.")
    return True


import torch
from torch import nn

import utils


class _ZeroScoreModel(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.calls = []

    def forward(self, x, t):
        self.calls.append((x.detach().clone(), t.detach().clone(), self.training))
        return torch.zeros_like(x)


class _ConstantScoreModel(nn.Module):
    def __init__(self, value=0.25, in_channels=3):
        super().__init__()
        self.value = value
        self.in_channels = in_channels
        self.calls = []

    def forward(self, x, t):
        self.calls.append((x.detach().clone(), t.detach().clone(), self.training))
        return torch.ones_like(x) * self.value


def test_vp_sde_sample():
    torch.manual_seed(0)

    device = torch.device("cpu")
    num_samples = 2
    image_size = 4
    num_steps = 3
    eps = 0.5
    channels = 3
    beta_start = 0.1
    beta_end = 0.5

    model = _ZeroScoreModel(in_channels=channels)
    model.train()

    original_randn = torch.randn
    original_randn_like = torch.randn_like

    try:
        torch.randn = lambda *size, **kwargs: torch.ones(*size, **kwargs) * 0.5
        torch.randn_like = lambda x: torch.zeros_like(x)

        samples, trajectories = utils.vp_sde_sample(
            model=model,
            num_samples=num_samples,
            image_size=image_size,
            num_steps=num_steps,
            eps=eps,
            device=device,
            beta_start=beta_start,
            beta_end=beta_end,
        )
    finally:
        torch.randn = original_randn
        torch.randn_like = original_randn_like

    expected_shape = (num_samples, channels, image_size, image_size)

    if not isinstance(samples, torch.Tensor):
        print("❌ FAIL: vp_sde_sample should return a tensor as first output.")
        return False

    if samples.shape != expected_shape:
        print("❌ FAIL: vp_sde_sample output has an unexpected shape.")
        return False

    if samples.device != device:
        print("❌ FAIL: vp_sde_sample output is on an unexpected device.")
        return False

    if not torch.isfinite(samples).all():
        print("❌ FAIL: vp_sde_sample output should contain only finite values.")
        return False

    if samples.requires_grad:
        print("❌ FAIL: vp_sde_sample should run without tracking gradients.")
        return False

    if not isinstance(trajectories, list):
        print("❌ FAIL: vp_sde_sample should return a list of intermediate samples.")
        return False

    if len(trajectories) != num_steps + 1:
        print("❌ FAIL: vp_sde_sample should store the initial sample and one sample per reverse step.")
        return False

    for item in trajectories:
        if not isinstance(item, torch.Tensor):
            print("❌ FAIL: Each stored trajectory item should be a tensor.")
            return False

        if item.shape != expected_shape:
            print("❌ FAIL: A stored trajectory item has an unexpected shape.")
            return False

        if item.device.type != "cpu":
            print("❌ FAIL: Stored trajectory items should be moved to CPU.")
            return False

        if not torch.isfinite(item).all():
            print("❌ FAIL: Stored trajectory items should contain only finite values.")
            return False

        if item.requires_grad:
            print("❌ FAIL: Stored trajectory items should not require gradients.")
            return False

    if not torch.allclose(trajectories[0], torch.ones(expected_shape) * 0.5, atol=1e-6):
        print("❌ FAIL: First trajectory item should contain the initial noise sample.")
        return False

    if len(model.calls) != num_steps:
        print("❌ FAIL: vp_sde_sample should call the model once per reverse step.")
        return False

    expected_times = torch.linspace(1.0, eps, steps=num_steps) * 999

    for i, expected_t in enumerate(expected_times):
        called_x, called_t, called_training = model.calls[i]

        if called_x.shape != expected_shape:
            print("❌ FAIL: vp_sde_sample passed an unexpected image tensor shape to the model.")
            return False

        if called_t.shape != (num_samples,):
            print("❌ FAIL: vp_sde_sample passed an unexpected time tensor shape to the model.")
            return False

        if not torch.allclose(called_t.cpu(), torch.ones(num_samples) * expected_t.cpu(), atol=1e-6):
            print("❌ FAIL: vp_sde_sample passed unexpected time values to the model.")
            return False

        if called_training:
            print("❌ FAIL: vp_sde_sample should evaluate the model in evaluation mode.")
            return False

    if not model.training:
        print("❌ FAIL: vp_sde_sample should restore the previous model mode.")
        return False

    # Deterministic value check with zero score and zero reverse noise.
    # Expected Euler-Maruyama reverse update:
    # x <- x - (drift - diffusion^2 * score) * dt + diffusion * sqrt(dt) * noise
    expected_value = 0.5
    timesteps = torch.linspace(1.0, eps, steps=num_steps)
    delta_t = (timesteps[0] - timesteps[1]).item()

    for time_value in timesteps:
        beta_t = beta_start + (beta_end - beta_start) * time_value.item()
        expected_value = expected_value * (1.0 + 0.5 * beta_t * delta_t)

    expected = torch.ones(expected_shape) * expected_value

    if not torch.allclose(samples.cpu(), expected, atol=1e-6):
        max_error = (samples.cpu() - expected).abs().max().item()
        print(f"❌ FAIL: vp_sde_sample produced unexpected deterministic values. Max error: {max_error:.2e}")
        return False

    if not torch.allclose(trajectories[-1], samples.cpu(), atol=1e-6):
        print("❌ FAIL: Final trajectory item should match the returned samples.")
        return False

    # Check score and stochastic term with controlled values.
    model = _ConstantScoreModel(value=0.25, in_channels=channels)
    model.eval()

    original_randn = torch.randn
    original_randn_like = torch.randn_like

    try:
        torch.randn = lambda *size, **kwargs: torch.ones(*size, **kwargs) * 0.5
        torch.randn_like = lambda x: torch.ones_like(x) * 0.1

        scored_samples, scored_trajectories = utils.vp_sde_sample(
            model=model,
            num_samples=num_samples,
            image_size=image_size,
            num_steps=2,
            eps=eps,
            device=device,
            beta_start=beta_start,
            beta_end=beta_end,
        )
    finally:
        torch.randn = original_randn
        torch.randn_like = original_randn_like

    expected_value = 0.5
    timesteps = torch.linspace(1.0, eps, steps=2)
    delta_t = (timesteps[0] - timesteps[1]).item()

    for time_value in timesteps:
        beta_t = beta_start + (beta_end - beta_start) * time_value.item()
        drift = -0.5 * beta_t * expected_value
        diffusion = beta_t ** 0.5
        score = 0.25
        reverse_noise = 0.1

        expected_value = (
            expected_value
            - (drift - beta_t * score) * delta_t
            + diffusion * (delta_t ** 0.5) * reverse_noise
        )

    expected_scored = torch.ones(expected_shape) * expected_value

    if not torch.allclose(scored_samples.cpu(), expected_scored, atol=1e-6):
        max_error = (scored_samples.cpu() - expected_scored).abs().max().item()
        print(f"❌ FAIL: vp_sde_sample does not correctly use score or reverse noise. Max error: {max_error:.2e}")
        return False

    if model.training:
        print("❌ FAIL: vp_sde_sample should preserve evaluation mode.")
        return False

    if len(scored_trajectories) != 3:
        print("❌ FAIL: vp_sde_sample stores an unexpected number of trajectory items.")
        return False

    # Check grayscale compatibility.
    gray_model = _ZeroScoreModel(in_channels=1)

    original_randn = torch.randn
    original_randn_like = torch.randn_like

    try:
        torch.randn = lambda *size, **kwargs: torch.zeros(*size, **kwargs)
        torch.randn_like = lambda x: torch.zeros_like(x)

        gray_samples, gray_trajectories = utils.vp_sde_sample(
            model=gray_model,
            num_samples=3,
            image_size=8,
            num_steps=2,
            eps=eps,
            device=device,
            beta_start=beta_start,
            beta_end=beta_end,
        )
    finally:
        torch.randn = original_randn
        torch.randn_like = original_randn_like

    if gray_samples.shape != (3, 1, 8, 8):
        print("❌ FAIL: vp_sde_sample should use the model input channel count.")
        return False

    if len(gray_trajectories) != 3:
        print("❌ FAIL: vp_sde_sample stores an unexpected number of grayscale trajectory items.")
        return False

    if torch.cuda.is_available():
        device = torch.device("cuda")

        cuda_model = _ZeroScoreModel(in_channels=channels).to(device)

        cuda_samples, cuda_trajectories = utils.vp_sde_sample(
            model=cuda_model,
            num_samples=1,
            image_size=4,
            num_steps=2,
            eps=eps,
            device=device,
            beta_start=beta_start,
            beta_end=beta_end,
        )

        if cuda_samples.device.type != "cuda":
            print("❌ FAIL: vp_sde_sample CUDA output is on an unexpected device.")
            return False

        for item in cuda_trajectories:
            if item.device.type != "cpu":
                print("❌ FAIL: CUDA trajectory items should be moved to CPU.")
                return False

    print("✅ PASS: vp_sde_sample seems to be correct.")
    return True