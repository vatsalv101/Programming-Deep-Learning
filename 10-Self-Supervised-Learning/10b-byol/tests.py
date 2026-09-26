import inspect
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from torch import nn

import data
import models
import pretrain


def _collect_transform_objects(obj, seen=None):
    if seen is None:
        seen = set()

    if id(obj) in seen:
        return []
    seen.add(id(obj))

    objects = [obj]

    if hasattr(obj, "transforms"):
        for child in obj.transforms:
            objects.extend(_collect_transform_objects(child, seen))

    if hasattr(obj, "transform"):
        objects.extend(_collect_transform_objects(obj.transform, seen))

    return objects


def _flatten_transforms(transform):
    """
    Recursively flattens torchvision transform containers.
    """
    flat = []

    if hasattr(transform, "transforms"):
        for child in transform.transforms:
            flat.extend(_flatten_transforms(child))
    else:
        flat.append(transform)

    return flat


def _flatten_transforms(transform):
    """
    Recursively returns all transforms, including container transforms
    such as Compose and RandomApply.
    """
    flat = [transform]

    if hasattr(transform, "transforms"):
        for child in transform.transforms:
            flat.extend(_flatten_transforms(child))

    return flat


def _find_transform(flat_transforms, transform_type):
    for transform in flat_transforms:
        if isinstance(transform, transform_type):
            return transform
    return None


def _find_random_apply_containing(flat_transforms, transform_type):
    for transform in flat_transforms:
        if isinstance(transform, transforms.RandomApply):
            children = []
            for child in transform.transforms:
                children.extend(_flatten_transforms(child))

            if any(isinstance(child, transform_type) for child in children):
                return transform

    return None


def _check_close(value, expected, name, atol=1e-6):
    if abs(float(value) - float(expected)) > atol:
        print(f"❌ FAIL: Expected {name} = {expected}, got {value}.")
        return False
    return True


def _check_tuple_close(value, expected, name, atol=1e-6):
    if len(value) != len(expected):
        print(f"❌ FAIL: Expected {name} = {expected}, got {value}.")
        return False

    for actual_value, expected_value in zip(value, expected):
        if abs(float(actual_value) - float(expected_value)) > atol:
            print(f"❌ FAIL: Expected {name} = {expected}, got {value}.")
            return False

    return True


def _check_two_view_transform_pipeline(transform, blur_probability, solarization_probability, name):
    flat_transforms = _flatten_transforms(transform)

    crop = _find_transform(flat_transforms, transforms.RandomResizedCrop)
    flip = _find_transform(flat_transforms, transforms.RandomHorizontalFlip)
    color_jitter_apply = _find_random_apply_containing(flat_transforms, transforms.ColorJitter)
    color_jitter = _find_transform(flat_transforms, transforms.ColorJitter)
    grayscale = _find_transform(flat_transforms, transforms.RandomGrayscale)
    blur_apply = _find_random_apply_containing(flat_transforms, transforms.GaussianBlur)
    blur = _find_transform(flat_transforms, transforms.GaussianBlur)
    solarize = _find_transform(flat_transforms, transforms.RandomSolarize)
    to_image = _find_transform(flat_transforms, transforms.ToImage)
    to_dtype = _find_transform(flat_transforms, transforms.ToDtype)
    normalize = _find_transform(flat_transforms, transforms.Normalize)

    if crop is None:
        print(f"❌ FAIL: {name} should include RandomResizedCrop.")
        return False

    if crop.size != (32, 32) and crop.size != 32:
        print(f"❌ FAIL: {name} should crop/resize to 32.")
        return False

    if not _check_tuple_close(crop.scale, (0.08, 1.0), f"{name} crop scale"):
        return False

    if not _check_tuple_close(crop.ratio, (3 / 4, 4 / 3), f"{name} crop ratio"):
        return False

    if flip is None:
        print(f"❌ FAIL: {name} should include RandomHorizontalFlip.")
        return False

    if not _check_close(flip.p, 0.5, f"{name} flip probability"):
        return False

    if color_jitter_apply is None or color_jitter is None:
        print(f"❌ FAIL: {name} should include RandomApply([ColorJitter(...)], p=0.8).")
        return False

    if not _check_close(color_jitter_apply.p, 0.8, f"{name} color jitter probability"):
        return False

    expected_color_jitter = {
        "brightness": (0.6, 1.4),
        "contrast": (0.6, 1.4),
        "saturation": (0.8, 1.2),
        "hue": (-0.1, 0.1),
    }

    for attribute, expected_value in expected_color_jitter.items():
        actual_value = getattr(color_jitter, attribute)
        if not _check_tuple_close(actual_value, expected_value, f"{name} ColorJitter.{attribute}"):
            return False

    if grayscale is None:
        print(f"❌ FAIL: {name} should include RandomGrayscale.")
        return False

    if not _check_close(grayscale.p, 0.2, f"{name} grayscale probability"):
        return False

    if blur_apply is None or blur is None:
        print(f"❌ FAIL: {name} should include RandomApply([GaussianBlur(...)], p={blur_probability}).")
        return False

    if not _check_close(blur_apply.p, blur_probability, f"{name} blur probability"):
        return False

    kernel_size = blur.kernel_size
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if tuple(kernel_size) != (3, 3):
        print(f"❌ FAIL: {name} GaussianBlur kernel size should be 3.")
        return False

    if not _check_tuple_close(blur.sigma, (0.1, 2.0), f"{name} GaussianBlur sigma"):
        return False

    if solarize is None:
        print(f"❌ FAIL: {name} should include RandomSolarize.")
        return False

    if not _check_close(solarize.threshold, 0.5, f"{name} solarization threshold"):
        return False

    if not _check_close(solarize.p, solarization_probability, f"{name} solarization probability"):
        return False

    if to_image is None:
        print(f"❌ FAIL: {name} should include ToImage().")
        return False

    if to_dtype is None:
        print(f"❌ FAIL: {name} should include ToDtype(torch.float32, scale=True).")
        return False

    if to_dtype.dtype != torch.float32:
        print(f"❌ FAIL: {name} ToDtype should use torch.float32.")
        return False

    if not to_dtype.scale:
        print(f"❌ FAIL: {name} ToDtype should use scale=True.")
        return False

    if normalize is None:
        print(f"❌ FAIL: {name} should include Normalize.")
        return False

    expected_mean = torch.tensor(data.CIFAR10_MEAN)
    expected_std = torch.tensor(data.CIFAR10_STD)
    actual_mean = torch.tensor(normalize.mean)
    actual_std = torch.tensor(normalize.std)

    if not torch.allclose(actual_mean, expected_mean, atol=1e-6):
        print(f"❌ FAIL: {name} should use CIFAR-10 mean for Normalize.")
        return False

    if not torch.allclose(actual_std, expected_std, atol=1e-6):
        print(f"❌ FAIL: {name} should use CIFAR-10 standard deviation for Normalize.")
        return False

    return True


def test_two_view_transform():
    """
    Tests whether TwoViewTransform follows the prescribed CIFAR-10
    adaptation of the augmentation recipe.
    """
    transform = data.TwoViewTransform()
    image = Image.fromarray(np.uint8(np.random.rand(32, 32, 3) * 255))

    if not hasattr(transform, "transform_1"):
        print("❌ FAIL: TwoViewTransform should define `self.transform_1`.")
        return False

    if not hasattr(transform, "transform_2"):
        print("❌ FAIL: TwoViewTransform should define `self.transform_2`.")
        return False

    try:
        view_1, view_2 = transform(image)
    except Exception as error:
        print(f"❌ FAIL: TwoViewTransform raised an error ({error}).")
        return False

    if not isinstance(view_1, torch.Tensor) or not isinstance(view_2, torch.Tensor):
        print("❌ FAIL: TwoViewTransform should return two tensors.")
        return False

    if view_1.shape != (3, 32, 32) or view_2.shape != (3, 32, 32):
        print("❌ FAIL: Each view should have shape (3, 32, 32).")
        return False

    if view_1.dtype != torch.float32 or view_2.dtype != torch.float32:
        print("❌ FAIL: Views should be float32 tensors.")
        return False

    if not torch.isfinite(view_1).all() or not torch.isfinite(view_2).all():
        print("❌ FAIL: Views should contain only finite values.")
        return False

    if not _check_two_view_transform_pipeline(
        transform.transform_1,
        blur_probability=1.0,
        solarization_probability=0.0,
        name="transform_1",
    ):
        return False

    if not _check_two_view_transform_pipeline(
        transform.transform_2,
        blur_probability=0.1,
        solarization_probability=0.2,
        name="transform_2",
    ):
        return False

    print("✅ PASS: TwoViewTransform looks correct.")
    return True


def _count_linear_layers(module):
    """
    Counts Linear layers inside a module.
    """
    return sum(1 for layer in module.modules() if isinstance(layer, nn.Linear))


def test_byol():
    """
    Tests the prescribed BYOL interface without prescribing the encoder architecture.
    """
    torch.manual_seed(0)

    try:
        model = models.BYOL()
    except Exception as error:
        print(f"❌ FAIL: Could not instantiate BYOL ({error}).")
        return False

    required_attributes = [
        "online_encoder",
        "online_projector",
        "online_predictor",
        "target_encoder",
        "target_projector",
    ]

    for attribute in required_attributes:
        if not hasattr(model, attribute):
            print(f"❌ FAIL: BYOL is missing required attribute `{attribute}`.")
            return False

    if _count_linear_layers(model.online_projector) < 2:
        print("❌ FAIL: `online_projector` should be an MLP with at least two Linear layers.")
        return False

    if _count_linear_layers(model.online_predictor) < 2:
        print("❌ FAIL: `online_predictor` should be an MLP with at least two Linear layers.")
        return False

    if _count_linear_layers(model.target_projector) < 2:
        print("❌ FAIL: `target_projector` should be an MLP with at least two Linear layers.")
        return False

    for parameter in model.target_encoder.parameters():
        if parameter.requires_grad:
            print("❌ FAIL: All target encoder parameters should have `requires_grad = False`.")
            return False

    for parameter in model.target_projector.parameters():
        if parameter.requires_grad:
            print("❌ FAIL: All target projector parameters should have `requires_grad = False`.")
            return False

    for parameter in model.online_encoder.parameters():
        if not parameter.requires_grad:
            print("❌ FAIL: Online encoder parameters should be trainable.")
            return False

    for parameter in model.online_projector.parameters():
        if not parameter.requires_grad:
            print("❌ FAIL: Online projector parameters should be trainable.")
            return False

    for parameter in model.online_predictor.parameters():
        if not parameter.requires_grad:
            print("❌ FAIL: Online predictor parameters should be trainable.")
            return False

    for key, online_tensor in model.online_encoder.state_dict().items():
        target_tensor = model.target_encoder.state_dict()[key]
        if not torch.allclose(online_tensor, target_tensor):
            print("❌ FAIL: Target encoder should start as a copy of the online encoder.")
            return False

    for key, online_tensor in model.online_projector.state_dict().items():
        target_tensor = model.target_projector.state_dict()[key]
        if not torch.allclose(online_tensor, target_tensor):
            print("❌ FAIL: Target projector should start as a copy of the online projector.")
            return False

    x = torch.randn(4, 3, 32, 32)

    try:
        online_output = model.online_forward(x)
    except Exception as error:
        print(f"❌ FAIL: `online_forward(x)` raised an error ({error}).")
        return False

    if not isinstance(online_output, tuple) or len(online_output) != 3:
        print("❌ FAIL: `online_forward(x)` should return `(features, projections, predictions)`.")
        return False

    features, projections, predictions = online_output

    if features.ndim != 2 or features.shape[0] != x.shape[0]:
        print("❌ FAIL: Online features should have shape (batch_size, feature_dim).")
        return False

    if projections.ndim != 2 or projections.shape[0] != x.shape[0]:
        print("❌ FAIL: Online projections should have shape (batch_size, projection_dim).")
        return False

    if predictions.ndim != 2 or predictions.shape[0] != x.shape[0]:
        print("❌ FAIL: Online predictions should have shape (batch_size, projection_dim).")
        return False

    if projections.shape != predictions.shape:
        print("❌ FAIL: Online projections and predictions should have the same shape.")
        return False

    try:
        target_output = model.target_forward(x)
    except Exception as error:
        print(f"❌ FAIL: `target_forward(x)` raised an error ({error}).")
        return False

    if not isinstance(target_output, tuple) or len(target_output) != 2:
        print("❌ FAIL: `target_forward(x)` should return `(features, projections)`.")
        return False

    target_features, target_projections = target_output

    if target_features.shape != features.shape:
        print("❌ FAIL: Target features should have the same shape as online features.")
        return False

    if target_projections.shape != projections.shape:
        print("❌ FAIL: Target projections should have the same shape as online projections.")
        return False

    try:
        encoded = model(x)
    except Exception as error:
        print(f"❌ FAIL: `model(x)` raised an error ({error}).")
        return False

    if encoded.shape != features.shape:
        print("❌ FAIL: `model(x)` should return online encoder features.")
        return False

    print("✅ PASS: BYOL class looks correct.")
    return True


def test_negative_cosine_similarity():
    """
    Tests the negative cosine similarity loss.
    """
    source = inspect.getsource(pretrain.negative_cosine_similarity)

    for forbidden in [
        "F.cosine_similarity",
        "torch.nn.functional.cosine_similarity",
        "nn.functional.cosine_similarity",
        ".cosine_similarity",
    ]:
        if forbidden in source:
            print("❌ FAIL: Implement the cosine-similarity formula manually instead of using cosine_similarity().")
            return False

    torch.manual_seed(0)

    p = torch.randn(4, 8, requires_grad=True)
    z = torch.randn(4, 8, requires_grad=True)

    try:
        loss = pretrain.negative_cosine_similarity(p, z)
    except Exception as error:
        print(f"❌ FAIL: negative_cosine_similarity() raised an error ({error}).")
        return False

    if loss.shape != ():
        print(
            "❌ FAIL: negative_cosine_similarity() should return a scalar tensor, "
            f"got shape {loss.shape}."
        )
        return False

    expected = 2.0 - 2.0 * (
        torch.nn.functional.normalize(p, dim=1)
        * torch.nn.functional.normalize(z.detach(), dim=1)
    ).sum(dim=1)
    expected = expected.mean()

    if not torch.allclose(loss, expected, atol=1e-6, rtol=1e-6):
        print("❌ FAIL: negative_cosine_similarity() returned the wrong value.")
        return False

    try:
        loss.backward()
    except Exception as error:
        print(
            "❌ FAIL: Backpropagation through negative_cosine_similarity() "
            f"failed ({error})."
        )
        return False

    if p.grad is None:
        print("❌ FAIL: The online prediction p should receive gradients.")
        return False

    if z.grad is not None:
        print("❌ FAIL: The target projection z should be detached from gradients.")
        return False

    p_same = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    z_same = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    p_opposite = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    z_opposite = torch.tensor([[-1.0, 0.0], [0.0, -1.0]])

    same_loss = pretrain.negative_cosine_similarity(p_same, z_same)
    opposite_loss = pretrain.negative_cosine_similarity(p_opposite, z_opposite)

    if abs(same_loss.item()) > 1e-6:
        print("❌ FAIL: Identical normalized vectors should produce loss 0.")
        return False

    if abs(opposite_loss.item() - 4.0) > 1e-6:
        print("❌ FAIL: Opposite normalized vectors should produce loss 4.")
        return False

    print("✅ PASS: Negative cosine similarity is correct.")
    return True


def test_ema_update():
    """
    Tests the exponential moving-average update for parameters and buffers.
    """
    torch.manual_seed(0)

    online = nn.BatchNorm1d(3)
    target = nn.BatchNorm1d(3)

    with torch.no_grad():
        online.weight.copy_(torch.tensor([1.0, 2.0, 3.0]))
        target.weight.copy_(torch.tensor([10.0, 20.0, 30.0]))

        online.bias.copy_(torch.tensor([4.0, 5.0, 6.0]))
        target.bias.copy_(torch.tensor([40.0, 50.0, 60.0]))

        online.running_mean.copy_(torch.tensor([1.0, 2.0, 3.0]))
        target.running_mean.copy_(torch.tensor([10.0, 20.0, 30.0]))

        online.running_var.copy_(torch.tensor([4.0, 5.0, 6.0]))
        target.running_var.copy_(torch.tensor([40.0, 50.0, 60.0]))

        online.num_batches_tracked.fill_(7)
        target.num_batches_tracked.fill_(21)

    for parameter in target.parameters():
        parameter.requires_grad = False

    online_weight_before = online.weight.detach().clone()
    online_bias_before = online.bias.detach().clone()
    online_mean_before = online.running_mean.detach().clone()
    online_var_before = online.running_var.detach().clone()
    online_batches_before = online.num_batches_tracked.detach().clone()

    target_weight_before = target.weight.detach().clone()
    target_bias_before = target.bias.detach().clone()
    target_mean_before = target.running_mean.detach().clone()
    target_var_before = target.running_var.detach().clone()

    tau = 0.75

    original_no_grad = pretrain.torch.no_grad

    class NoGradTracker:
        def __init__(self, original_context):
            self.original_context = original_context
            self.entered = False

        def __call__(self):
            context = self.original_context()

            tracker = self

            class WrappedContext:
                def __enter__(self_inner):
                    tracker.entered = True
                    return context.__enter__()

                def __exit__(self_inner, exc_type, exc_value, traceback):
                    return context.__exit__(exc_type, exc_value, traceback)

            return WrappedContext()

    tracker = NoGradTracker(original_no_grad)
    pretrain.torch.no_grad = tracker

    try:
        pretrain.update_moving_average(online, target, tau=tau)
    except Exception as error:
        print(f"❌ FAIL: update_moving_average() raised an error ({error}).")
        pretrain.torch.no_grad = original_no_grad
        return False
    finally:
        pretrain.torch.no_grad = original_no_grad

    if not tracker.entered:
        print("❌ FAIL: update_moving_average() should use `with torch.no_grad():`.")
        return False

    expected_weight = tau * target_weight_before + (1.0 - tau) * online_weight_before
    expected_bias = tau * target_bias_before + (1.0 - tau) * online_bias_before
    expected_mean = tau * target_mean_before + (1.0 - tau) * online_mean_before
    expected_var = tau * target_var_before + (1.0 - tau) * online_var_before

    if not torch.allclose(target.weight, expected_weight):
        print("❌ FAIL: update_moving_average() used the wrong EMA formula for parameters.")
        return False

    if not torch.allclose(target.bias, expected_bias):
        print("❌ FAIL: update_moving_average() used the wrong EMA formula for parameters.")
        return False

    if not torch.allclose(target.running_mean, expected_mean):
        print("❌ FAIL: update_moving_average() should also EMA-update floating-point buffers.")
        return False

    if not torch.allclose(target.running_var, expected_var):
        print("❌ FAIL: update_moving_average() should also EMA-update floating-point buffers.")
        return False

    if not torch.equal(target.num_batches_tracked, online_batches_before):
        print("❌ FAIL: non-floating-point buffers should be copied from online to target.")
        return False

    if not torch.allclose(online.weight, online_weight_before):
        print("❌ FAIL: update_moving_average() should not modify online parameters.")
        return False

    if not torch.allclose(online.running_mean, online_mean_before):
        print("❌ FAIL: update_moving_average() should not modify online buffers.")
        return False

    if any(parameter.requires_grad for parameter in target.parameters()):
        print("❌ FAIL: Target parameters should remain frozen.")
        return False

    print("✅ PASS: EMA target update is correct.")
    return True
