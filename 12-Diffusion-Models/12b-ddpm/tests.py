import torch
from torch import nn

import utils

def test_ddpm_schedule():
    torch.manual_seed(0)

    schedule = utils.precompute_ddpm_schedule(
        num_timesteps=4,
        beta_start=0.1,
        beta_end=0.4,
        device=None,
    )

    required_keys = {
        "betas",
        "alphas",
        "alpha_bars",
        "sqrt_alpha_bars",
        "sqrt_one_minus_alpha_bars",
    }

    if not isinstance(schedule, dict):
        print("❌ FAIL: Schedule should be returned as a dictionary.")
        return False

    if set(schedule.keys()) != required_keys:
        print("❌ FAIL: Schedule contains unexpected keys.")
        return False

    for key in required_keys:
        if not isinstance(schedule[key], torch.Tensor):
            print(f"❌ FAIL: Schedule entry '{key}' should be a tensor.")
            return False

        if schedule[key].shape != (4,):
            print(f"❌ FAIL: Schedule entry '{key}' has an unexpected shape.")
            return False

        if not torch.isfinite(schedule[key]).all():
            print(f"❌ FAIL: Schedule entry '{key}' should contain only finite values.")
            return False

    expected_betas = torch.tensor([0.1, 0.2, 0.3, 0.4])
    expected_alphas = torch.tensor([0.9, 0.8, 0.7, 0.6])
    expected_alpha_bars = torch.tensor([0.9, 0.72, 0.504, 0.3024])
    expected_sqrt_alpha_bars = torch.tensor([0.9486833, 0.8485281, 0.7099296, 0.5499091])
    expected_sqrt_one_minus_alpha_bars = torch.tensor([0.3162278, 0.5291503, 0.7042727, 0.8352245])

    checks = [
        ("betas", expected_betas),
        ("alphas", expected_alphas),
        ("alpha_bars", expected_alpha_bars),
        ("sqrt_alpha_bars", expected_sqrt_alpha_bars),
        ("sqrt_one_minus_alpha_bars", expected_sqrt_one_minus_alpha_bars),
    ]

    for key, expected in checks:
        if not torch.allclose(schedule[key], expected, atol=1e-6):
            max_error = (schedule[key] - expected).abs().max().item()
            print(f"❌ FAIL: Schedule entry '{key}' has incorrect values. Max error: {max_error:.2e}")
            return False

    if not torch.all(schedule["betas"][1:] > schedule["betas"][:-1]):
        print("❌ FAIL: Beta values should increase over time.")
        return False

    if not torch.all(schedule["alpha_bars"][1:] < schedule["alpha_bars"][:-1]):
        print("❌ FAIL: Cumulative noise scaling values should decrease over time.")
        return False

    if not torch.all(schedule["sqrt_alpha_bars"] >= 0):
        print("❌ FAIL: Square-root entries should be non-negative.")
        return False

    if not torch.all(schedule["sqrt_one_minus_alpha_bars"] >= 0):
        print("❌ FAIL: Square-root entries should be non-negative.")
        return False

    consistency = schedule["sqrt_alpha_bars"] ** 2 + schedule["sqrt_one_minus_alpha_bars"] ** 2

    if not torch.allclose(consistency, torch.ones_like(consistency), atol=1e-6):
        print("❌ FAIL: Schedule entries are internally inconsistent.")
        return False

    if torch.cuda.is_available():
        device = torch.device("cuda")
        cuda_schedule = utils.precompute_ddpm_schedule(
            num_timesteps=4,
            beta_start=0.1,
            beta_end=0.4,
            device=device,
        )

        for key in required_keys:
            if cuda_schedule[key].device.type != "cuda":
                print(f"❌ FAIL: Schedule entry '{key}' is on an unexpected device.")
                return False

    print("✅ PASS: DDPM schedule seems to be correct.")
    return True


def test_q_sample():
    torch.manual_seed(0)

    x0 = torch.tensor(
        [
            [[[0.0, 1.0],
              [2.0, 3.0]]],
            [[[4.0, 5.0],
              [6.0, 7.0]]],
        ]
    )

    noise = torch.tensor(
        [
            [[[10.0, 11.0],
              [12.0, 13.0]]],
            [[[20.0, 21.0],
              [22.0, 23.0]]],
        ]
    )

    t = torch.tensor([0, 2], dtype=torch.long)

    sqrt_alpha_bars = torch.tensor([1.0, 0.5, 0.25])
    sqrt_one_minus_alpha_bars = torch.tensor([0.0, 0.5, 0.75])

    xt = utils.q_sample(
        x0=x0,
        t=t,
        noise=noise,
        sqrt_alpha_bars=sqrt_alpha_bars,
        sqrt_one_minus_alpha_bars=sqrt_one_minus_alpha_bars,
    )

    expected = torch.tensor(
        [
            [[[0.0, 1.0],
              [2.0, 3.0]]],
            [[[16.0, 17.0],
              [18.0, 19.0]]],
        ]
    )

    if xt.shape != x0.shape:
        print("❌ FAIL: q_sample output should have the same shape as x0.")
        return False

    if xt.dtype != x0.dtype:
        print("❌ FAIL: q_sample output should have the same dtype as x0.")
        return False

    if not torch.isfinite(xt).all():
        print("❌ FAIL: q_sample output should contain only finite values.")
        return False

    if not torch.allclose(xt, expected, atol=1e-6):
        max_error = (xt - expected).abs().max().item()
        print(f"❌ FAIL: q_sample values are incorrect. Max error: {max_error:.2e}")
        return False

    # Check that different batch elements can use different time indices.
    x0_batch = torch.ones(3, 1, 2, 2)
    noise_batch = torch.ones_like(x0_batch) * 10.0
    t_batch = torch.tensor([0, 1, 2], dtype=torch.long)

    xt_batch = utils.q_sample(
        x0=x0_batch,
        t=t_batch,
        noise=noise_batch,
        sqrt_alpha_bars=sqrt_alpha_bars,
        sqrt_one_minus_alpha_bars=sqrt_one_minus_alpha_bars,
    )

    expected_batch = torch.tensor([1.0, 5.5, 7.75]).view(3, 1, 1, 1).expand_as(x0_batch)

    if not torch.allclose(xt_batch, expected_batch, atol=1e-6):
        max_error = (xt_batch - expected_batch).abs().max().item()
        print(f"❌ FAIL: q_sample does not handle per-sample time indices correctly. Max error: {max_error:.2e}")
        return False

    # Clean-image limit.
    xt_clean = utils.q_sample(
        x0=x0,
        t=torch.zeros(2, dtype=torch.long),
        noise=noise,
        sqrt_alpha_bars=torch.ones(3),
        sqrt_one_minus_alpha_bars=torch.zeros(3),
    )

    if not torch.allclose(xt_clean, x0, atol=1e-6):
        print("❌ FAIL: q_sample should preserve x0 in the clean-image limit.")
        return False

    # Pure-noise limit.
    xt_noise = utils.q_sample(
        x0=x0,
        t=torch.zeros(2, dtype=torch.long),
        noise=noise,
        sqrt_alpha_bars=torch.zeros(3),
        sqrt_one_minus_alpha_bars=torch.ones(3),
    )

    if not torch.allclose(xt_noise, noise, atol=1e-6):
        print("❌ FAIL: q_sample should return noise in the pure-noise limit.")
        return False

    xt_again = utils.q_sample(
        x0=x0,
        t=t,
        noise=noise,
        sqrt_alpha_bars=sqrt_alpha_bars,
        sqrt_one_minus_alpha_bars=sqrt_one_minus_alpha_bars,
    )

    if not torch.allclose(xt, xt_again, atol=1e-6):
        print("❌ FAIL: q_sample should be deterministic for fixed inputs.")
        return False

    x0_train = x0.clone().requires_grad_(True)

    xt_train = utils.q_sample(
        x0=x0_train,
        t=t,
        noise=noise,
        sqrt_alpha_bars=sqrt_alpha_bars,
        sqrt_one_minus_alpha_bars=sqrt_one_minus_alpha_bars,
    )

    loss = xt_train.square().mean()
    loss.backward()

    if x0_train.grad is None:
        print("❌ FAIL: q_sample should support backpropagation through x0.")
        return False

    if not torch.isfinite(x0_train.grad).all():
        print("❌ FAIL: q_sample gradients should contain only finite values.")
        return False

    print("✅ PASS: q_sample seems to be correct.")
    return True


class _ZeroNoiseModel(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.calls = []

    def forward(self, x, t):
        self.calls.append((x.detach().clone(), t.detach().clone(), self.training))
        return torch.zeros_like(x)


class _ConstantNoiseModel(nn.Module):
    def __init__(self, value=1.0, in_channels=3):
        super().__init__()
        self.value = value
        self.in_channels = in_channels
        self.calls = []

    def forward(self, x, t):
        self.calls.append((x.detach().clone(), t.detach().clone(), self.training))
        return torch.ones_like(x) * self.value


def _small_test_schedule(device=None):
    return {
        "betas": torch.tensor([0.1, 0.2, 0.3], device=device),
        "alphas": torch.tensor([0.9, 0.8, 0.7], device=device),
        "alpha_bars": torch.tensor([0.9, 0.72, 0.504], device=device),
        "sqrt_alpha_bars": torch.tensor([0.9486833, 0.8485281, 0.7099296], device=device),
        "sqrt_one_minus_alpha_bars": torch.tensor([0.3162278, 0.5291503, 0.7042727], device=device),
    }


def test_ddpm_p_sample():
    torch.manual_seed(0)

    device = torch.device("cpu")
    schedule = _small_test_schedule(device=device)

    batch_size = 2
    channels = 3
    height = 4
    width = 4

    xt = torch.randn(batch_size, channels, height, width, device=device)
    t = torch.full((batch_size,), 0, dtype=torch.long, device=device)

    model = _ZeroNoiseModel(in_channels=channels)
    model.train()

    x_prev = utils.ddpm_p_sample(
        model=model,
        xt=xt,
        t=t,
        t_index=0,
        schedule=schedule,
    )

    expected = xt / torch.sqrt(schedule["alphas"][0])

    if x_prev.shape != xt.shape:
        print("❌ FAIL: ddpm_p_sample output should have the same shape as xt.")
        return False

    if x_prev.dtype != xt.dtype:
        print("❌ FAIL: ddpm_p_sample output should have the same dtype as xt.")
        return False

    if x_prev.device != xt.device:
        print("❌ FAIL: ddpm_p_sample output should be on the same device as xt.")
        return False

    if not torch.isfinite(x_prev).all():
        print("❌ FAIL: ddpm_p_sample output should contain only finite values.")
        return False

    if not torch.allclose(x_prev, expected, atol=1e-6):
        max_error = (x_prev - expected).abs().max().item()
        print(f"❌ FAIL: ddpm_p_sample gives incorrect values for the final reverse step. Max error: {max_error:.2e}")
        return False

    if len(model.calls) != 1:
        print("❌ FAIL: ddpm_p_sample should call the model exactly once.")
        return False

    called_x, called_t, called_training = model.calls[0]

    if not torch.allclose(called_x, xt, atol=1e-6):
        print("❌ FAIL: ddpm_p_sample passed an unexpected image tensor to the model.")
        return False

    if not torch.equal(called_t, t):
        print("❌ FAIL: ddpm_p_sample passed an unexpected time tensor to the model.")
        return False

    if called_training:
        print("❌ FAIL: ddpm_p_sample should evaluate the model in evaluation mode.")
        return False

    if not model.training:
        print("❌ FAIL: ddpm_p_sample should restore the previous model mode.")
        return False

    # Test a non-final reverse step with controlled noise.
    model = _ConstantNoiseModel(value=0.5, in_channels=channels)
    model.eval()

    xt = torch.ones(batch_size, channels, height, width, device=device)
    t = torch.full((batch_size,), 2, dtype=torch.long, device=device)

    original_randn_like = torch.randn_like

    try:
        torch.randn_like = lambda x: torch.ones_like(x) * 0.25

        x_prev = utils.ddpm_p_sample(
            model=model,
            xt=xt,
            t=t,
            t_index=2,
            schedule=schedule,
        )
    finally:
        torch.randn_like = original_randn_like

    expected_value = 1.0076474
    expected = torch.ones_like(xt) * expected_value

    if not torch.allclose(x_prev, expected, atol=1e-5):
        max_error = (x_prev - expected).abs().max().item()
        print(f"❌ FAIL: ddpm_p_sample gives incorrect values for an intermediate reverse step. Max error: {max_error:.2e}")
        return False

    if model.training:
        print("❌ FAIL: ddpm_p_sample should preserve evaluation mode.")
        return False

    # Check that gradients are not tracked.
    xt_grad = torch.randn(batch_size, channels, height, width, requires_grad=True)
    t_grad = torch.zeros(batch_size, dtype=torch.long)

    model = _ZeroNoiseModel(in_channels=channels)
    x_prev_grad = utils.ddpm_p_sample(
        model=model,
        xt=xt_grad,
        t=t_grad,
        t_index=0,
        schedule=_small_test_schedule(),
    )

    if x_prev_grad.requires_grad:
        print("❌ FAIL: ddpm_p_sample should run without tracking gradients.")
        return False

    print("✅ PASS: ddpm_p_sample seems to be correct.")
    return True


def test_ddpm_sample():
    torch.manual_seed(0)

    device = torch.device("cpu")
    num_samples = 2
    image_size = 4
    num_timesteps = 3
    channels = 3

    schedule = _small_test_schedule(device=device)

    model = _ZeroNoiseModel(in_channels=channels)
    model.train()

    original_randn = torch.randn
    original_randn_like = torch.randn_like

    try:
        torch.randn = lambda *size, **kwargs: torch.ones(*size, **kwargs) * 0.5
        torch.randn_like = lambda x: torch.zeros_like(x)

        samples, trajectories = utils.ddpm_sample(
            model=model,
            num_samples=num_samples,
            image_size=image_size,
            num_timesteps=num_timesteps,
            device=device,
            schedule=schedule,
        )
    finally:
        torch.randn = original_randn
        torch.randn_like = original_randn_like

    expected_shape = (num_samples, channels, image_size, image_size)

    if not isinstance(samples, torch.Tensor):
        print("❌ FAIL: ddpm_sample should return a tensor as first output.")
        return False

    if samples.shape != expected_shape:
        print("❌ FAIL: ddpm_sample output has an unexpected shape.")
        return False

    if samples.device != device:
        print("❌ FAIL: ddpm_sample output is on an unexpected device.")
        return False

    if not torch.isfinite(samples).all():
        print("❌ FAIL: ddpm_sample output should contain only finite values.")
        return False

    if not isinstance(trajectories, list):
        print("❌ FAIL: ddpm_sample should return a list of intermediate samples.")
        return False

    if len(trajectories) != num_timesteps + 1:
        print("❌ FAIL: ddpm_sample should store the initial sample and one sample per reverse step.")
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

    if len(model.calls) != num_timesteps:
        print("❌ FAIL: ddpm_sample should call the model once per reverse step.")
        return False

    expected_times = [2, 1, 0]

    for i, expected_t in enumerate(expected_times):
        _, called_t, called_training = model.calls[i]

        if called_t.shape != (num_samples,):
            print("❌ FAIL: ddpm_sample passed an unexpected time tensor shape to the model.")
            return False

        if not torch.equal(called_t.cpu(), torch.full((num_samples,), expected_t, dtype=torch.long)):
            print("❌ FAIL: ddpm_sample used an unexpected reverse time order.")
            return False

        if called_training:
            print("❌ FAIL: ddpm_sample should evaluate the model in evaluation mode.")
            return False

    if not model.training:
        print("❌ FAIL: ddpm_sample should restore the previous model mode.")
        return False

    # With zero predicted noise and zero reverse noise, the result is deterministic here.
    expected_value = 0.5
    for idx in [2, 1, 0]:
        expected_value = expected_value / torch.sqrt(schedule["alphas"][idx]).item()

    expected = torch.ones(expected_shape) * expected_value

    if not torch.allclose(samples.cpu(), expected, atol=1e-5):
        max_error = (samples.cpu() - expected).abs().max().item()
        print(f"❌ FAIL: ddpm_sample produced unexpected sample values. Max error: {max_error:.2e}")
        return False

    if not torch.allclose(trajectories[-1], samples.cpu(), atol=1e-6):
        print("❌ FAIL: Final trajectory item should match the returned samples.")
        return False

    # Check grayscale compatibility.
    gray_model = _ZeroNoiseModel(in_channels=1)

    original_randn = torch.randn
    original_randn_like = torch.randn_like

    try:
        torch.randn = lambda *size, **kwargs: torch.zeros(*size, **kwargs)
        torch.randn_like = lambda x: torch.zeros_like(x)

        gray_samples, gray_trajectories = utils.ddpm_sample(
            model=gray_model,
            num_samples=3,
            image_size=8,
            num_timesteps=num_timesteps,
            device=device,
            schedule=schedule,
        )
    finally:
        torch.randn = original_randn
        torch.randn_like = original_randn_like

    if gray_samples.shape != (3, 1, 8, 8):
        print("❌ FAIL: ddpm_sample should use the model input channel count.")
        return False

    if len(gray_trajectories) != num_timesteps + 1:
        print("❌ FAIL: ddpm_sample stores an unexpected number of grayscale trajectory items.")
        return False

    # Check that gradients are not tracked.
    model = _ZeroNoiseModel(in_channels=channels)

    samples_no_grad, _ = utils.ddpm_sample(
        model=model,
        num_samples=1,
        image_size=4,
        num_timesteps=1,
        device=device,
        schedule=schedule,
    )

    if samples_no_grad.requires_grad:
        print("❌ FAIL: ddpm_sample should run without tracking gradients.")
        return False

    print("✅ PASS: ddpm_sample seems to be correct.")
    return True