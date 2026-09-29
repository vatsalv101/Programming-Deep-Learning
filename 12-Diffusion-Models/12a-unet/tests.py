import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from tqdm.auto import tqdm
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import models


def _reference_time_embedding(t, dim):
    """
    Reference implementation for

        omega_i = exp(-(i / (h - 1)) * log(10000))
        emb(t) = [sin(t omega_0), ..., sin(t omega_{h-1}),
                  cos(t omega_0), ..., cos(t omega_{h-1})]

    If dim is odd, append one zero.
    """
    t = t.float()
    h = dim // 2

    freqs = torch.exp(
        -torch.arange(h, dtype=torch.float32, device=t.device)
        / (h - 1)
        * torch.log(torch.tensor(10000.0, device=t.device))
    )

    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

    return emb


def test_time_embedding():
    torch.manual_seed(0)

    dim = 16
    emb_layer = models.SinusoidalTimeEmbedding(dim=dim)

    t = torch.tensor([0, 1, 10, 999], dtype=torch.long)
    emb = emb_layer(t)
    expected = _reference_time_embedding(t, dim)

    if emb.shape != (4, dim):
        print("❌ FAIL: Time embedding should have shape (B, dim).")
        return False

    if not torch.allclose(emb, expected, atol=1e-6):
        max_error = (emb - expected).abs().max().item()
        print(f"❌ FAIL: Time embedding values are incorrect. Max error: {max_error:.2e}")
        return False

    # At t = 0, sin-part should be 0 and cos-part should be 1.
    if not torch.allclose(emb[0, : dim // 2], torch.zeros(dim // 2), atol=1e-6):
        print("❌ FAIL: For t=0, the sine part should be zero.")
        return False

    if not torch.allclose(emb[0, dim // 2 :], torch.ones(dim // 2), atol=1e-6):
        print("❌ FAIL: For t=0, the cosine part should be one.")
        return False

    t_float = torch.tensor([0.01, 0.5, 1.0], dtype=torch.float32)
    emb_float = emb_layer(t_float)
    expected_float = _reference_time_embedding(t_float, dim)

    if emb_float.shape != (3, dim):
        print("❌ FAIL: Continuous time embedding should have shape (B, dim).")
        return False

    if not torch.allclose(emb_float, expected_float, atol=1e-6):
        max_error = (emb_float - expected_float).abs().max().item()
        print(f"❌ FAIL: Continuous time embedding values are incorrect. Max error: {max_error:.2e}")
        return False

    # This checks that continuous values are not rounded internally.
    emb_half = emb_layer(torch.tensor([0.5]))
    emb_zero = emb_layer(torch.tensor([0.0]))
    emb_one = emb_layer(torch.tensor([1.0]))

    if torch.allclose(emb_half, emb_zero, atol=1e-6) or torch.allclose(emb_half, emb_one, atol=1e-6):
        print("❌ FAIL: Continuous time values seem to be rounded or discretized.")
        return False

    odd_dim = 15
    odd_layer = models.SinusoidalTimeEmbedding(dim=odd_dim)

    t_odd = torch.tensor([0, 3, 7], dtype=torch.long)
    emb_odd = odd_layer(t_odd)
    expected_odd = _reference_time_embedding(t_odd, odd_dim)

    if emb_odd.shape != (3, odd_dim):
        print("❌ FAIL: Odd-dimensional time embedding should have shape (B, dim).")
        return False

    if not torch.allclose(emb_odd, expected_odd, atol=1e-6):
        max_error = (emb_odd - expected_odd).abs().max().item()
        print(f"❌ FAIL: Odd-dimensional time embedding values are incorrect. Max error: {max_error:.2e}")
        return False

    if not torch.allclose(emb_odd[:, -1], torch.zeros(3), atol=1e-6):
        print("❌ FAIL: For odd dim, the final padding entry should be zero.")
        return False

    emb_again = emb_layer(t)

    if not torch.allclose(emb, emb_again, atol=1e-6):
        print("❌ FAIL: Time embedding should be deterministic.")
        return False

    if not torch.isfinite(emb).all():
        print("❌ FAIL: Time embedding should contain only finite values.")
        return False

    print("✅ PASS: Time embedding seems to be correct.")
    return True


def _num_parameters(module):
    return sum(p.numel() for p in module.parameters())


def _conv2d_params(in_channels, out_channels, kernel_size, bias=True):
    return out_channels * in_channels * kernel_size * kernel_size + (out_channels if bias else 0)


def _linear_params(in_features, out_features, bias=True):
    return out_features * in_features + (out_features if bias else 0)


def _groupnorm_params(channels):
    return 2 * channels


def _expected_resblock_params(in_channels, out_channels, time_dim):
    total = 0

    # conv1: GN -> SiLU -> Conv2d
    total += _groupnorm_params(in_channels)
    total += _conv2d_params(in_channels, out_channels, kernel_size=3, bias=True)

    # time projection: SiLU -> Linear
    total += _linear_params(time_dim, out_channels, bias=True)

    # conv2: GN -> SiLU -> Conv2d
    total += _groupnorm_params(out_channels)
    total += _conv2d_params(out_channels, out_channels, kernel_size=3, bias=True)

    # skip connection
    if in_channels != out_channels:
        total += _conv2d_params(in_channels, out_channels, kernel_size=1, bias=True)

    return total


def _expected_downsample_params(channels):
    return _conv2d_params(channels, channels, kernel_size=4, bias=True)


def _expected_upsample_params(channels):
    return _conv2d_params(channels, channels, kernel_size=3, bias=True)


def _expected_unet_params(in_channels, base_channels, time_dim):
    C = base_channels
    D = in_channels
    T = time_dim

    total = 0

    # time embedding MLP
    total += _linear_params(T, 4 * T, bias=True)
    total += _linear_params(4 * T, T, bias=True)

    # initial convolution
    total += _conv2d_params(D, C, kernel_size=3, bias=True)

    # encoder
    total += _expected_resblock_params(C, C, T)
    total += _expected_downsample_params(C)

    total += _expected_resblock_params(C, 2 * C, T)
    total += _expected_downsample_params(2 * C)

    total += _expected_resblock_params(2 * C, 4 * C, T)
    total += _expected_downsample_params(4 * C)

    # bottleneck
    total += _expected_resblock_params(4 * C, 4 * C, T)

    # decoder
    total += _expected_upsample_params(4 * C)
    total += _expected_resblock_params(8 * C, 4 * C, T)

    total += _expected_upsample_params(4 * C)
    total += _expected_resblock_params(6 * C, 2 * C, T)

    total += _expected_upsample_params(2 * C)
    total += _expected_resblock_params(3 * C, C, T)

    # output block: GN -> SiLU -> Conv2d
    total += _groupnorm_params(C)
    total += _conv2d_params(C, D, kernel_size=3, bias=True)

    return total


def test_resblock():
    torch.manual_seed(0)

    batch_size = 4
    in_channels = 8
    out_channels = 16
    time_dim = 32
    groups = 4

    block = models.ResBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        time_dim=time_dim,
        groups=groups,
    )

    x = torch.randn(batch_size, in_channels, 16, 16)
    t_emb = torch.randn(batch_size, time_dim)

    out = block(x, t_emb)

    if out.shape != (batch_size, out_channels, 16, 16):
        print("❌ FAIL: ResBlock output has an unexpected shape.")
        return False

    if not torch.isfinite(out).all():
        print("❌ FAIL: ResBlock output should contain only finite values.")
        return False

    expected_params = _expected_resblock_params(in_channels, out_channels, time_dim)
    actual_params = _num_parameters(block)

    if actual_params != expected_params:
        print(
            f"❌ FAIL: ResBlock has an unexpected number of parameters. "
            f"Expected {expected_params}, got {actual_params}."
        )
        return False

    conv_layers = [m for m in block.modules() if isinstance(m, nn.Conv2d)]
    norm_layers = [m for m in block.modules() if isinstance(m, nn.GroupNorm)]
    linear_layers = [m for m in block.modules() if isinstance(m, nn.Linear)]

    if len(conv_layers) != 3:
        print("❌ FAIL: ResBlock has an unexpected convolutional structure.")
        return False

    if len(norm_layers) != 2:
        print("❌ FAIL: ResBlock has an unexpected normalization structure.")
        return False

    if len(linear_layers) != 1:
        print("❌ FAIL: ResBlock has an unexpected time projection structure.")
        return False

    t_emb_1 = torch.zeros(batch_size, time_dim)
    t_emb_2 = torch.ones(batch_size, time_dim)

    out_1 = block(x, t_emb_1)
    out_2 = block(x, t_emb_2)

    if torch.allclose(out_1, out_2, atol=1e-6):
        print("❌ FAIL: ResBlock output should depend on the time embedding.")
        return False

    # Also check the case where no channel projection is needed.
    same_block = models.ResBlock(
        in_channels=in_channels,
        out_channels=in_channels,
        time_dim=time_dim,
        groups=groups,
    )

    same_out = same_block(x, t_emb)

    if same_out.shape != x.shape:
        print("❌ FAIL: ResBlock output has an unexpected shape when channels match.")
        return False

    expected_same_params = _expected_resblock_params(in_channels, in_channels, time_dim)
    actual_same_params = _num_parameters(same_block)

    if actual_same_params != expected_same_params:
        print(
            f"❌ FAIL: ResBlock has an unexpected number of parameters when channels match. "
            f"Expected {expected_same_params}, got {actual_same_params}."
        )
        return False

    # Check that non-divisible channel counts are handled safely.
    try:
        fallback_block = models.ResBlock(
            in_channels=3,
            out_channels=5,
            time_dim=time_dim,
            groups=groups,
        )
        fallback_x = torch.randn(batch_size, 3, 8, 8)
        fallback_t = torch.randn(batch_size, time_dim)
        fallback_out = fallback_block(fallback_x, fallback_t)
    except Exception:
        print("❌ FAIL: ResBlock should handle this channel configuration.")
        return False

    if fallback_out.shape != (batch_size, 5, 8, 8):
        print("❌ FAIL: ResBlock output has an unexpected shape for this channel configuration.")
        return False

    # Check backward pass.
    x_train = torch.randn(batch_size, in_channels, 16, 16, requires_grad=True)
    t_train = torch.randn(batch_size, time_dim)
    out_train = block(x_train, t_train)
    loss = out_train.square().mean()
    loss.backward()

    if x_train.grad is None:
        print("❌ FAIL: ResBlock should support backpropagation.")
        return False

    if not torch.isfinite(x_train.grad).all():
        print("❌ FAIL: ResBlock gradients should contain only finite values.")
        return False

    print("✅ PASS: ResBlock seems to be correct.")
    return True


def test_downsample():
    torch.manual_seed(0)

    batch_size = 4
    channels = 8

    layer = models.Downsample(channels)

    x = torch.randn(batch_size, channels, 32, 32)
    out = layer(x)

    if out.shape != (batch_size, channels, 16, 16):
        print("❌ FAIL: Downsample output has an unexpected shape.")
        return False

    if not torch.isfinite(out).all():
        print("❌ FAIL: Downsample output should contain only finite values.")
        return False

    expected_params = _expected_downsample_params(channels)
    actual_params = _num_parameters(layer)

    if actual_params != expected_params:
        print(
            f"❌ FAIL: Downsample has an unexpected number of parameters. "
            f"Expected {expected_params}, got {actual_params}."
        )
        return False

    conv_layers = [m for m in layer.modules() if isinstance(m, nn.Conv2d)]

    if len(conv_layers) != 1:
        print("❌ FAIL: Downsample has an unexpected structure.")
        return False

    conv = conv_layers[0]

    if conv.kernel_size != (4, 4) or conv.stride != (2, 2) or conv.padding != (1, 1):
        print("❌ FAIL: Downsample uses unexpected convolution settings.")
        return False

    x_train = torch.randn(batch_size, channels, 32, 32, requires_grad=True)
    loss = layer(x_train).square().mean()
    loss.backward()

    if x_train.grad is None or not torch.isfinite(x_train.grad).all():
        print("❌ FAIL: Downsample should support backpropagation.")
        return False

    print("✅ PASS: Downsample seems to be correct.")
    return True


def test_upsample():
    torch.manual_seed(0)

    batch_size = 4
    channels = 8

    layer = models.Upsample(channels)

    x = torch.randn(batch_size, channels, 16, 16)
    out = layer(x)

    if out.shape != (batch_size, channels, 32, 32):
        print("❌ FAIL: Upsample output has an unexpected shape.")
        return False

    if not torch.isfinite(out).all():
        print("❌ FAIL: Upsample output should contain only finite values.")
        return False

    expected_params = _expected_upsample_params(channels)
    actual_params = _num_parameters(layer)

    if actual_params != expected_params:
        print(
            f"❌ FAIL: Upsample has an unexpected number of parameters. "
            f"Expected {expected_params}, got {actual_params}."
        )
        return False

    has_upsample = any(isinstance(m, nn.Upsample) for m in layer.modules())
    conv_layers = [m for m in layer.modules() if isinstance(m, nn.Conv2d)]

    if not has_upsample:
        print("❌ FAIL: Upsample has an unexpected structure.")
        return False

    if len(conv_layers) != 1:
        print("❌ FAIL: Upsample has an unexpected convolutional structure.")
        return False

    conv = conv_layers[0]

    if conv.kernel_size != (3, 3) or conv.padding != (1, 1):
        print("❌ FAIL: Upsample uses unexpected convolution settings.")
        return False

    x_train = torch.randn(batch_size, channels, 16, 16, requires_grad=True)
    loss = layer(x_train).square().mean()
    loss.backward()

    if x_train.grad is None or not torch.isfinite(x_train.grad).all():
        print("❌ FAIL: Upsample should support backpropagation.")
        return False

    print("✅ PASS: Upsample seems to be correct.")
    return True


def test_unet():
    torch.manual_seed(0)

    batch_size = 2
    in_channels = 3
    base_channels = 8
    time_dim = 32
    groups = 4

    model = models.UNet(
        in_channels=in_channels,
        base_channels=base_channels,
        time_dim=time_dim,
        groups=groups,
    )

    model.eval()

    x = torch.randn(batch_size, in_channels, 32, 32)
    t = torch.randint(0, 1000, (batch_size,))

    resblock_shapes = []
    downsample_shapes = []
    upsample_shapes = []
    time_shapes = []

    hooks = []

    def make_resblock_hook():
        def hook(module, inputs, output):
            resblock_shapes.append((tuple(inputs[0].shape), tuple(output.shape)))
            time_shapes.append(tuple(inputs[1].shape))
        return hook

    def make_downsample_hook():
        def hook(module, inputs, output):
            downsample_shapes.append(tuple(output.shape))
        return hook

    def make_upsample_hook():
        def hook(module, inputs, output):
            upsample_shapes.append(tuple(output.shape))
        return hook

    for module in model.modules():
        if isinstance(module, models.ResBlock):
            hooks.append(module.register_forward_hook(make_resblock_hook()))
        elif isinstance(module, models.Downsample):
            hooks.append(module.register_forward_hook(make_downsample_hook()))
        elif isinstance(module, models.Upsample):
            hooks.append(module.register_forward_hook(make_upsample_hook()))

    with torch.no_grad():
        out = model(x, t)

    for hook in hooks:
        hook.remove()

    if out.shape != x.shape:
        print("❌ FAIL: UNet output should have the same shape as the input.")
        return False

    if not torch.isfinite(out).all():
        print("❌ FAIL: UNet output should contain only finite values.")
        return False

    expected_params = _expected_unet_params(in_channels, base_channels, time_dim)
    actual_params = _num_parameters(model)

    if actual_params != expected_params:
        print(
            f"❌ FAIL: UNet has an unexpected number of parameters. "
            f"Expected {expected_params}, got {actual_params}."
        )
        return False

    C = base_channels
    B = batch_size

    expected_resblock_shapes = [
        ((B, C, 32, 32), (B, C, 32, 32)),
        ((B, C, 16, 16), (B, 2 * C, 16, 16)),
        ((B, 2 * C, 8, 8), (B, 4 * C, 8, 8)),
        ((B, 4 * C, 4, 4), (B, 4 * C, 4, 4)),
        ((B, 8 * C, 8, 8), (B, 4 * C, 8, 8)),
        ((B, 6 * C, 16, 16), (B, 2 * C, 16, 16)),
        ((B, 3 * C, 32, 32), (B, C, 32, 32)),
    ]

    expected_downsample_shapes = [
        (B, C, 16, 16),
        (B, 2 * C, 8, 8),
        (B, 4 * C, 4, 4),
    ]

    expected_upsample_shapes = [
        (B, 4 * C, 8, 8),
        (B, 4 * C, 16, 16),
        (B, 2 * C, 32, 32),
    ]

    if len(resblock_shapes) != 7:
        print("❌ FAIL: UNet has an unexpected number of residual blocks.")
        return False

    if len(downsample_shapes) != 3:
        print("❌ FAIL: UNet has an unexpected encoder structure.")
        return False

    if len(upsample_shapes) != 3:
        print("❌ FAIL: UNet has an unexpected decoder structure.")
        return False

    if resblock_shapes != expected_resblock_shapes:
        print("❌ FAIL: UNet feature flow is incorrect.")
        return False

    if downsample_shapes != expected_downsample_shapes:
        print("❌ FAIL: UNet encoder feature sizes are incorrect.")
        return False

    if upsample_shapes != expected_upsample_shapes:
        print("❌ FAIL: UNet decoder feature sizes are incorrect.")
        return False

    if not all(shape == (B, time_dim) for shape in time_shapes):
        print("❌ FAIL: UNet time conditioning has an unexpected shape.")
        return False

    # Check that the time input actually affects intermediate features.
    def collect_resblock_outputs(t_value):
        outputs = []
        local_hooks = []

        def make_hook():
            def hook(module, inputs, output):
                outputs.append(output.detach().clone())
            return hook

        for module in model.modules():
            if isinstance(module, models.ResBlock):
                local_hooks.append(module.register_forward_hook(make_hook()))

        with torch.no_grad():
            _ = model(x, t_value)

        for hook in local_hooks:
            hook.remove()

        return outputs

    t_zero = torch.zeros(batch_size, dtype=torch.long)
    t_late = torch.full((batch_size,), 999, dtype=torch.long)

    outputs_zero = collect_resblock_outputs(t_zero)
    outputs_late = collect_resblock_outputs(t_late)

    if len(outputs_zero) != 7 or len(outputs_late) != 7:
        print("❌ FAIL: UNet conditioning behavior is unexpected.")
        return False

    time_changes_features = any(
        not torch.allclose(a, b, atol=1e-6)
        for a, b in zip(outputs_zero, outputs_late)
    )

    if not time_changes_features:
        print("❌ FAIL: UNet output should depend on the time input.")
        return False

    # Check grayscale compatibility.
    gray_model = models.UNet(
        in_channels=1,
        base_channels=base_channels,
        time_dim=time_dim,
        groups=groups,
    )

    gray_x = torch.randn(batch_size, 1, 32, 32)
    gray_t = torch.randint(0, 1000, (batch_size,))

    with torch.no_grad():
        gray_out = gray_model(gray_x, gray_t)

    if gray_out.shape != gray_x.shape:
        print("❌ FAIL: UNet should handle a different number of input channels.")
        return False

    expected_gray_params = _expected_unet_params(1, base_channels, time_dim)
    actual_gray_params = _num_parameters(gray_model)

    if actual_gray_params != expected_gray_params:
        print(
            f"❌ FAIL: UNet has an unexpected number of parameters for this input setting. "
            f"Expected {expected_gray_params}, got {actual_gray_params}."
        )
        return False

    # Check backward pass.
    model.train()
    x_train = torch.randn(batch_size, in_channels, 32, 32)
    t_train = torch.randint(0, 1000, (batch_size,))
    target = torch.randn_like(x_train)

    pred = model(x_train, t_train)
    loss = F.mse_loss(pred, target)
    loss.backward()

    grads = [
        p.grad
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]

    if len(grads) == 0:
        print("❌ FAIL: UNet should support backpropagation.")
        return False

    if not all(torch.isfinite(g).all() for g in grads):
        print("❌ FAIL: UNet gradients should contain only finite values.")
        return False

    print("✅ PASS: UNet seems to be correct.")
    return True


def simple_mnist_ddpm():

    """
    Trains a DDPM noise prediction model on MNIST images.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")
    
    dset = torchvision.datasets.MNIST(root="../data", train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize((32, 32)), torchvision.transforms.Normalize((0.5,), (0.5,))]))
    dloader = DataLoader(dset, batch_size=64, shuffle=True, num_workers=2)

    model = models.UNet(in_channels=1, base_channels=16, time_dim=32, groups=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    betas = torch.linspace(1e-4, 2e-2, 1000, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    sqrt_alpha_bars = torch.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    epochs = 2
    pbar = tqdm(range(epochs), desc="Training DDPM")
    for epoch in pbar:
        model.train()
        train_loss = 0.0
        train_samples = 0

        for x, _ in tqdm(dloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            x = x.to(device, non_blocking=True)
            t = torch.randint(0, 1000, (x.shape[0],), device=device, dtype=torch.long)
            noise = torch.randn_like(x)
            
            sqrt_alpha_bars_t = sqrt_alpha_bars.gather(0, t).view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_bars_t = sqrt_one_minus_alpha_bars.gather(0, t).view(-1, 1, 1, 1)
            
            xt = sqrt_alpha_bars_t * x + sqrt_one_minus_alpha_bars_t * noise

            pred_noise = model(xt, t)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.shape[0]
            train_samples += x.shape[0]

        train_loss /= max(train_samples, 1)

        tqdm.write(f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.4f}")

    print("Training complete. Generating samples...")

    with torch.no_grad():
        model.eval()

        xt = torch.randn(16, model.in_channels, 32, 32, device=device)
        for t_index in tqdm(reversed(range(1000)), desc="Generating samples"):
            t = torch.full((16,), t_index, device=device)
            eps = model(xt, t)
            mu = (1 / alphas[t_index].sqrt()) * (xt - (betas[t_index] / sqrt_one_minus_alpha_bars[t_index]) * eps)
            xt = mu
            if t_index > 0:
                noise = torch.randn_like(xt)
                sigma_t = sqrt_one_minus_alpha_bars[t_index - 1] / sqrt_one_minus_alpha_bars[t_index] * betas[t_index].sqrt()
                xt = mu + sigma_t * noise  

        images = (xt + 1) / 2
        images = images.clamp(0.0, 1.0).detach().cpu()

        images = images.repeat(1, 3, 1, 1)
        images = images.permute(0, 2, 3, 1)
        images = (255 * images).clamp(0, 255).to(torch.uint8).numpy()

        fig = make_subplots(
            rows=4,
            cols=4,
            horizontal_spacing=0.03,
            vertical_spacing=0.03,
        )

        for index in range(16):
            row = index // 4 + 1
            col = index % 4 + 1
            fig.add_trace(go.Image(z=images[index]), row=row, col=col)

        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        fig.update_layout(
            height=160 * 4,
            width=160 * 4,
            margin=dict(l=20, r=20, t=20, b=20),
        )

        fig.show()