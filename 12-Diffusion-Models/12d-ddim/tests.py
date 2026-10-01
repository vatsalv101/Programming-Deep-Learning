import torch
import torch.nn as nn
import utils

class _ZeroNoiseModel(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
    def forward(self, x, t):
        return torch.zeros_like(x)

class _ConstantNoiseModel(nn.Module):
    def __init__(self, value=0.5, in_channels=3):
        super().__init__()
        self.value = value
        self.in_channels = in_channels
    def forward(self, x, t):
        return torch.ones_like(x) * self.value

def test_get_ddim_timesteps():
    timesteps = utils.get_ddim_timesteps(num_timesteps=1000, num_ddim_steps=10, step_type="linear")
    
    if timesteps is None or not isinstance(timesteps, torch.Tensor):
        print("❌ FAIL: get_ddim_timesteps should return a 1D torch.Tensor.")
        return False
    if len(timesteps) != 10:
        print(f"❌ FAIL: Expected 10 timesteps, got {len(timesteps)}.")
        return False
    if timesteps[0] != 999:
        print(f"❌ FAIL: First timestep should be 999, got {timesteps[0].item()}.")
        return False
    if not torch.all(timesteps[:-1] > timesteps[1:]):
        print("❌ FAIL: Timesteps must be strictly descending.")
        return False
        
    print("✅ PASS: get_ddim_timesteps is correct!")
    return True

def test_ddim_p_sample():
    torch.manual_seed(42)
    device = torch.device("cpu")
    schedule = utils.precompute_ddpm_schedule(100, device=device)
    
    model = _ConstantNoiseModel(value=0.2)
    model.eval()
    xt = torch.randn(2, 3, 8, 8, device=device)
    
    # Test Deterministic behavior (eta = 0.0)
    x_prev_1 = utils.ddim_p_sample(model, xt, 50, 40, schedule, eta=0.0)
    x_prev_2 = utils.ddim_p_sample(model, xt, 50, 40, schedule, eta=0.0)
    
    if x_prev_1 is None:
        print("❌ FAIL: ddim_p_sample returned None.")
        return False
    if not torch.allclose(x_prev_1, x_prev_2, atol=1e-6):
        print("❌ FAIL: DDIM step with eta=0.0 must be strictly deterministic!")
        return False
        
    print("✅ PASS: ddim_p_sample is correct!")
    return True

def test_ddim_sample():
    torch.manual_seed(0)
    device = torch.device("cpu")
    schedule = utils.precompute_ddpm_schedule(100, device=device)
    model = _ZeroNoiseModel(in_channels=3)
    model.eval()
    
    samples, trajectories = utils.ddim_sample(
        model, num_samples=2, image_size=8, num_timesteps=100, num_ddim_steps=5, device=device, schedule=schedule, eta=0.0
    )
    
    if samples is None:
        print("❌ FAIL: ddim_sample returned None.")
        return False
    if samples.shape != (2, 3, 8, 8):
        print(f"❌ FAIL: Expected sample shape (2, 3, 8, 8), got {samples.shape}.")
        return False
    if len(trajectories) != 6: # Initial noise + 5 steps
        print(f"❌ FAIL: Expected 6 trajectory frames (initial + 5 steps), got {len(trajectories)}.")
        return False
        
    print("✅ PASS: ddim_sample is correct!")
    return True