import torch
from torch import nn

import models
import pretrain
import downstream


def _copy_parameters(module):
    """
    Returns detached copies of all parameters in a module.
    """
    return [
        parameter.detach().clone()
        for parameter in module.parameters()
    ]


def _parameters_changed(parameters_before, module):
    """
    Checks whether at least one parameter changed.
    """
    parameters_after = list(module.parameters())

    return any(
        not torch.allclose(before, after.detach())
        for before, after in zip(parameters_before, parameters_after)
    )


def _parameters_unchanged(parameters_before, module):
    """
    Checks whether all parameters stayed unchanged.
    """
    parameters_after = list(module.parameters())

    return all(
        torch.allclose(before, after.detach())
        for before, after in zip(parameters_before, parameters_after)
    )


def test_rotations():
    """
    Tests whether rotation labels and rotated images are created correctly.
    """
    torch.manual_seed(0)

    # First test apply_rotations(), since this is the core helper.
    x = torch.tensor(
        [[
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0]]
        ]]
    )

    x = x.repeat(4, 1, 1, 1)
    x_before = x.clone()
    labels = torch.tensor([0, 1, 2, 3])

    try:
        rotated_x = pretrain.apply_rotations(x, labels)
    except Exception as error:
        print(f"❌ FAIL: apply_rotations() raised an error ({error}).")
        return False

    if not torch.equal(x, x_before):
        print(
            "❌ FAIL: apply_rotations() should not modify the input tensor "
            "in-place."
        )
        return False

    expected = torch.tensor(
        [
            [[[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0],
              [7.0, 8.0, 9.0]]],

            [[[3.0, 6.0, 9.0],
              [2.0, 5.0, 8.0],
              [1.0, 4.0, 7.0]]],

            [[[9.0, 8.0, 7.0],
              [6.0, 5.0, 4.0],
              [3.0, 2.0, 1.0]]],

            [[[7.0, 4.0, 1.0],
              [8.0, 5.0, 2.0],
              [9.0, 6.0, 3.0]]],
        ]
    )

    if rotated_x.shape != expected.shape:
        print(
            "❌ FAIL: apply_rotations() returned images with the wrong shape: "
            f"got {rotated_x.shape}, expected {expected.shape}."
        )
        return False

    if not torch.equal(rotated_x, expected):
        print(
            "❌ FAIL: apply_rotations() does not correctly apply all four "
            "rotations."
        )
        return False

    # Test random rotations.
    x = torch.randn(8, 3, 32, 32)

    try:
        rotated_x, y = pretrain.make_random_rotations(x)
    except Exception as error:
        print(f"❌ FAIL: make_random_rotations() raised an error ({error}).")
        return False

    if rotated_x.shape != x.shape:
        print(
            "❌ FAIL: make_random_rotations() returned images with the wrong "
            f"shape: got {rotated_x.shape}, expected {x.shape}."
        )
        return False

    if y.shape != (8,):
        print(
            "❌ FAIL: make_random_rotations() returned labels with the wrong "
            f"shape: got {y.shape}, expected (8,)."
        )
        return False

    if y.dtype != torch.long:
        print(
            "❌ FAIL: make_random_rotations() should return rotation labels "
            "with dtype torch.long."
        )
        return False

    if (y < 0).any() or (y > 3).any():
        print(
            "❌ FAIL: make_random_rotations() should only return labels "
            "in {0, 1, 2, 3}."
        )
        return False

    expected_rotated_x = pretrain.apply_rotations(x, y)

    if not torch.allclose(rotated_x, expected_rotated_x):
        print(
            "❌ FAIL: make_random_rotations() returned images that do not match "
            "the returned rotation labels."
        )
        return False

    # Test deterministic rotations.
    try:
        rotated_x, y = pretrain.make_deterministic_rotations(x)
    except Exception as error:
        print(f"❌ FAIL: make_deterministic_rotations() raised an error ({error}).")
        return False

    expected_y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

    if not torch.equal(y.cpu(), expected_y):
        print(
            "❌ FAIL: make_deterministic_rotations() returned the wrong labels. "
            f"Got {y.cpu().tolist()}, expected {expected_y.tolist()}."
        )
        return False

    if rotated_x.shape != x.shape:
        print(
            "❌ FAIL: make_deterministic_rotations() returned images with the "
            f"wrong shape: got {rotated_x.shape}, expected {x.shape}."
        )
        return False

    expected_rotated_x = pretrain.apply_rotations(x, y)

    if not torch.allclose(rotated_x, expected_rotated_x):
        print(
            "❌ FAIL: make_deterministic_rotations() returned images that do "
            "not match the deterministic rotation labels."
        )
        return False

    print("✅ PASS: rotation functions are correct.")
    return True


def test_rotation_classifier():
    """
    Tests whether RotationClassifier behaves like a four-class classifier.
    """
    torch.manual_seed(0)

    try:
        model = models.RotationClassifier()
    except Exception as error:
        print(f"❌ FAIL: RotationClassifier could not be initialized ({error}).")
        return False

    if not isinstance(model, nn.Module):
        print("❌ FAIL: RotationClassifier must inherit from torch.nn.Module.")
        return False

    if not hasattr(model, "encoder"):
        print("❌ FAIL: RotationClassifier should store the encoder as self.encoder.")
        return False

    if not isinstance(model.encoder, models.CNNEncoder):
        print("❌ FAIL: RotationClassifier should create a CNNEncoder as self.encoder.")
        return False

    if not hasattr(model, "head"):
        print("❌ FAIL: RotationClassifier should store the prediction head as self.head.")
        return False

    linear_layers = [
        module
        for module in model.head.modules()
        if isinstance(module, nn.Linear)
    ]

    if len(linear_layers) != 2:
        print(
            "❌ FAIL: RotationClassifier.head should contain exactly two "
            "linear layers."
        )
        return False

    if linear_layers[0].in_features != model.encoder.out_dim:
        print(
            "❌ FAIL: The first linear layer in RotationClassifier.head should "
            "take encoder.out_dim input features."
        )
        return False

    if linear_layers[1].out_features != 4:
        print(
            "❌ FAIL: The second linear layer in RotationClassifier.head should "
            "output four logits."
        )
        return False

    x = torch.randn(4, 3, 32, 32)

    try:
        logits = model(x)
    except Exception as error:
        print(f"❌ FAIL: RotationClassifier.forward() raised an error ({error}).")
        return False

    if logits.shape != (4, 4):
        print(
            "❌ FAIL: RotationClassifier.forward() returned the wrong shape: "
            f"got {logits.shape}, expected (4, 4)."
        )
        return False

    if not torch.is_floating_point(logits):
        print("❌ FAIL: RotationClassifier.forward() should return floating point logits.")
        return False

    if not torch.isfinite(logits).all():
        print("❌ FAIL: RotationClassifier.forward() returned non-finite values.")
        return False

    row_sums = logits.sum(dim=1)

    if torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-3,
        rtol=1e-3,
    ):
        print(
            "❌ FAIL: RotationClassifier should return raw logits, not softmax "
            "probabilities."
        )
        return False

    model.zero_grad()
    loss = logits.mean()

    try:
        loss.backward()
    except Exception as error:
        print(f"❌ FAIL: Backpropagation through RotationClassifier failed ({error}).")
        return False

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if len(gradients) == 0:
        print("❌ FAIL: RotationClassifier has no trainable parameters.")
        return False

    if any(gradient is None for gradient in gradients):
        print("❌ FAIL: Some RotationClassifier parameters did not receive gradients.")
        return False

    if not all(torch.isfinite(gradient).all() for gradient in gradients):
        print("❌ FAIL: Some RotationClassifier gradients contain non-finite values.")
        return False

    print("✅ PASS: RotationClassifier is correct.")
    return True


def test_rotnet_training_step():
    """
    Tests whether one RotNet training step works correctly.
    """
    torch.manual_seed(0)

    class SpyRotationModel(nn.Module):
        """
        Small model that records its input and returns trainable logits.
        """
        def __init__(self):
            super().__init__()
            self.logits = nn.Parameter(torch.tensor([2.0, 0.0, -1.0, -2.0]))
            self.last_input = None
            self.last_logits = None

        def forward(self, x):
            self.last_input = x.detach().clone()
            logits = self.logits.unsqueeze(0).repeat(x.shape[0], 1)
            self.last_logits = logits.detach().clone()
            return logits

    model = SpyRotationModel()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-2,
        weight_decay=0.0,
    )

    loss_fn = nn.CrossEntropyLoss()

    x = torch.randn(8, 3, 32, 32)

    called_make_random_rotations = {"value": False}
    expected_y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

    def fake_make_random_rotations(x):
        called_make_random_rotations["value"] = True
        return x + 10.0, expected_y.to(x.device)

    original_make_random_rotations = pretrain.make_random_rotations
    pretrain.make_random_rotations = fake_make_random_rotations

    parameters_before = _copy_parameters(model)

    try:
        loss_value = pretrain.rotnet_training_step(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            x=x,
            device="cpu",
        )
    except Exception as error:
        print(f"❌ FAIL: rotnet_training_step() raised an error ({error}).")
        return False
    finally:
        pretrain.make_random_rotations = original_make_random_rotations

    pretrain.make_random_rotations = original_make_random_rotations

    if not called_make_random_rotations["value"]:
        print(
            "❌ FAIL: rotnet_training_step() should use make_random_rotations() "
            "to create rotated images and rotation labels."
        )
        return False

    if model.last_input is None:
        print("❌ FAIL: rotnet_training_step() did not call the model.")
        return False

    expected_model_input = x + 10.0

    if not torch.allclose(model.last_input, expected_model_input):
        print(
            "❌ FAIL: rotnet_training_step() should pass the rotated images "
            "returned by make_random_rotations() into the model."
        )
        return False

    expected_loss = loss_fn(model.last_logits, expected_y).item()

    if abs(loss_value - expected_loss) > 1e-5:
        print(
            "❌ FAIL: rotnet_training_step() should compute the loss using "
            "the rotation labels returned by make_random_rotations()."
        )
        return False

    if not isinstance(loss_value, float):
        print(
            "❌ FAIL: rotnet_training_step() should return the loss value "
            "as a Python float."
        )
        return False

    if not torch.isfinite(torch.tensor(loss_value)):
        print("❌ FAIL: rotnet_training_step() returned a non-finite loss.")
        return False

    if loss_value <= 0:
        print("❌ FAIL: rotnet_training_step() returned a non-positive loss.")
        return False

    if not _parameters_changed(parameters_before, model):
        print(
            "❌ FAIL: rotnet_training_step() did not update the model "
            "parameters."
        )
        return False

    print("✅ PASS: RotNet training step is correct.")
    return True


def test_downstream_classifier():
    """
    Tests whether DownstreamClassifier uses a frozen encoder and returns
    CIFAR-10 logits.
    """
    torch.manual_seed(0)

    try:
        encoder = models.CNNEncoder()
        model = models.DownstreamClassifier(encoder)
    except Exception as error:
        print(f"❌ FAIL: DownstreamClassifier could not be initialized ({error}).")
        return False

    if not isinstance(model, nn.Module):
        print("❌ FAIL: DownstreamClassifier must inherit from torch.nn.Module.")
        return False

    if not hasattr(model, "encoder"):
        print("❌ FAIL: DownstreamClassifier should store the encoder as self.encoder.")
        return False

    if not hasattr(model, "head"):
        print("❌ FAIL: DownstreamClassifier should store the classification head as self.head.")
        return False

    if model.encoder is not encoder:
        print(
            "❌ FAIL: DownstreamClassifier should use the encoder that was "
            "passed to its constructor."
        )
        return False

    x = torch.randn(4, 3, 32, 32)

    model.train()

    try:
        logits = model(x)
    except Exception as error:
        print(f"❌ FAIL: DownstreamClassifier.forward() raised an error ({error}).")
        return False

    if logits.shape != (4, 10):
        print(
            "❌ FAIL: DownstreamClassifier.forward() returned the wrong shape: "
            f"got {logits.shape}, expected (4, 10)."
        )
        return False

    if not torch.is_floating_point(logits):
        print("❌ FAIL: DownstreamClassifier.forward() should return floating point logits.")
        return False

    if not torch.isfinite(logits).all():
        print("❌ FAIL: DownstreamClassifier.forward() returned non-finite values.")
        return False

    if torch.all(logits >= 0):
        print(
            "❌ FAIL: DownstreamClassifier should return raw logits, not ReLU outputs."
        )
        return False

    row_sums = logits.sum(dim=1)

    if torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-3,
        rtol=1e-3,
    ):
        print(
            "❌ FAIL: DownstreamClassifier should return raw logits, not softmax "
            "probabilities."
        )
        return False

    if model.encoder.training:
        print(
            "❌ FAIL: DownstreamClassifier should keep the encoder in eval mode "
            "during the forward pass."
        )
        return False

    model.zero_grad()
    loss = logits.mean()

    try:
        loss.backward()
    except Exception as error:
        print(f"❌ FAIL: Backpropagation through DownstreamClassifier failed ({error}).")
        return False

    encoder_gradients = [
        parameter.grad
        for parameter in model.encoder.parameters()
    ]

    head_gradients = [
        parameter.grad
        for parameter in model.head.parameters()
        if parameter.requires_grad
    ]

    if any(gradient is not None for gradient in encoder_gradients):
        print(
            "❌ FAIL: DownstreamClassifier should not compute gradients for "
            "the encoder."
        )
        return False

    if len(head_gradients) == 0:
        print("❌ FAIL: DownstreamClassifier has no trainable head parameters.")
        return False

    if any(gradient is None for gradient in head_gradients):
        print(
            "❌ FAIL: The classification head should receive gradients during "
            "backpropagation."
        )
        return False

    if not all(torch.isfinite(gradient).all() for gradient in head_gradients):
        print("❌ FAIL: Some classification head gradients contain non-finite values.")
        return False

    print("✅ PASS: DownstreamClassifier is correct.")
    return True


def test_create_downstream_model_and_optimizer():
    """
    Tests whether the downstream model and optimizer are created correctly.
    """
    torch.manual_seed(0)

    try:
        rotnet = models.RotationClassifier()
        model, optimizer = downstream.create_downstream_model_and_optimizer(
            rotnet=rotnet,
            lr=1e-3,
            weight_decay=1e-4,
        )
    except Exception as error:
        print(
            "❌ FAIL: create_downstream_model_and_optimizer() raised an error "
            f"({error})."
        )
        return False

    if not isinstance(model, models.DownstreamClassifier):
        print(
            "❌ FAIL: create_downstream_model_and_optimizer() should return a "
            "DownstreamClassifier as the first output."
        )
        return False

    if not isinstance(optimizer, torch.optim.Optimizer):
        print(
            "❌ FAIL: create_downstream_model_and_optimizer() should return a "
            "PyTorch optimizer as the second output."
        )
        return False

    if model.encoder is not rotnet.encoder:
        print(
            "❌ FAIL: The downstream classifier should reuse the encoder from "
            "the pretrained RotationClassifier."
        )
        return False

    x = torch.randn(4, 3, 32, 32)

    try:
        logits = model(x)
    except Exception as error:
        print(
            "❌ FAIL: The returned downstream model raised an error during the "
            f"forward pass ({error})."
        )
        return False

    if logits.shape != (4, 10):
        print(
            "❌ FAIL: The returned downstream model has the wrong output shape: "
            f"got {logits.shape}, expected (4, 10)."
        )
        return False

    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    head_parameter_ids = {
        id(parameter)
        for parameter in model.head.parameters()
    }

    encoder_parameter_ids = {
        id(parameter)
        for parameter in model.encoder.parameters()
    }

    if optimizer_parameter_ids != head_parameter_ids:
        print(
            "❌ FAIL: The downstream optimizer should receive exactly the "
            "classification head parameters."
        )
        return False

    if len(optimizer_parameter_ids & encoder_parameter_ids) > 0:
        print(
            "❌ FAIL: The downstream optimizer should not receive encoder "
            "parameters."
        )
        return False

    print("✅ PASS: downstream model and optimizer are created correctly.")
    return True
