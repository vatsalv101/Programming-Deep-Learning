import torch
from torch import nn

import cnn


def _get_leaf_modules(model: nn.Module) -> list[nn.Module]:
    """
    Returns all leaf modules of a model.

    Args:
        model (nn.Module): Model to inspect.

    Returns:
        leaf_modules (list[nn.Module]): List of modules without children.
    """
    leaf_modules = []

    for module in model.modules():
        if module is model:
            continue
        if len(list(module.children())) == 0:
            leaf_modules.append(module)

    return leaf_modules


def test_simple_cnn():
    """
    Tests the CNN architecture, output shapes, and gradient flow.

    Returns: bool: True if the test passed, False otherwise.
    """
    torch.manual_seed(0)

    try:
        model = cnn.SimpleCNN()
    except Exception as error:
        print(f"❌ FAIL: SimpleCNN() could not be initialized ({error}).")
        return False

    if not isinstance(model, nn.Module):
        print("❌ FAIL: SimpleCNN must inherit from torch.nn.Module.")
        return False

    has_linear = any(
        isinstance(module, nn.Linear)
        for module in model.modules()
    )
    if not has_linear:
        print(
            "❌ FAIL: Your model does not contain an nn.Linear layer. "
            "The final classifier should map the flattened features to class logits."
        )
        return False

    x = torch.randn(16, 3, 32, 32)

    try:
        logits = model(x)
    except Exception as error:
        print(f"❌ FAIL: SimpleCNN.forward() raised an error ({error}).")
        return False

    if logits.shape != (16, 10):
        print(
            "❌ FAIL: SimpleCNN.forward() returned the wrong shape: "
            f"got {logits.shape}, expected (16, 10)."
        )
        return False

    if not torch.is_floating_point(logits):
        print("❌ FAIL: SimpleCNN.forward() should return floating point logits.")
        return False

    if not torch.isfinite(logits).all():
        print("❌ FAIL: SimpleCNN.forward() returned non-finite values.")
        return False

    leaf_modules = _get_leaf_modules(model)
    forbidden_final_modules = (
        nn.ReLU,
        nn.Sigmoid,
        nn.Softmax,
        nn.LogSoftmax,
        nn.Tanh,
    )

    if len(leaf_modules) > 0 and isinstance(leaf_modules[-1], forbidden_final_modules):
        print(
            "❌ FAIL: The last layer of your model seems to be an activation layer. "
            "The network should return raw logits."
        )
        return False

    if torch.all(logits >= 0):
        print(
            "❌ FAIL: The model output is non-negative for all tested logits. "
            "This suggests that a final activation such as ReLU may have been applied. "
            "The network should return raw logits."
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
            "❌ FAIL: The model output sums to 1 across classes. "
            "This suggests that a final softmax was applied. "
            "The network should return raw logits."
        )
        return False

    try:
        predicted_class = model.predict_class(x)
    except Exception as error:
        print(f"❌ FAIL: SimpleCNN.predict_class() raised an error ({error}).")
        return False

    if predicted_class.shape != (16,):
        print(
            "❌ FAIL: SimpleCNN.predict_class() returned the wrong shape: "
            f"got {predicted_class.shape}, expected (16,)."
        )
        return False

    if predicted_class.dtype != torch.int64:
        print(
            "❌ FAIL: SimpleCNN.predict_class() should return class indices "
            "with dtype torch.int64."
        )
        return False

    if (predicted_class < 0).any() or (predicted_class >= 10).any():
        print("❌ FAIL: SimpleCNN.predict_class() returned invalid class indices.")
        return False

    model.zero_grad()
    loss = logits.sum()

    try:
        loss.backward()
    except Exception as error:
        print(f"❌ FAIL: backpropagation through SimpleCNN failed ({error}).")
        return False

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if any(gradient is None for gradient in gradients):
        print("❌ FAIL: Some trainable parameters did not receive gradients.")
        return False

    if not all(torch.isfinite(gradient).all() for gradient in gradients):
        print("❌ FAIL: Some parameter gradients contain non-finite values.")
        return False

    has_batchnorm = any(
        isinstance(module, nn.BatchNorm2d)
        for module in model.modules()
    )

    if not has_batchnorm:
        print(
            "💡 SUGGESTION: Your model does not use nn.BatchNorm2d. "
            "Try adding batch normalization to improve training stability."
        )

    print("✅ PASS: SimpleCNN is correct.")
    return True


def _check_optimizer_hyperparameters(optimizer, lr, weight_decay):
    """
    Checks whether the optimizer is AdamW and uses the expected hyperparameters.
    """
    if not isinstance(optimizer, torch.optim.AdamW):
        print(
            "❌ FAIL: initialize() should return a torch.optim.AdamW optimizer "
            "as its second return value."
        )
        return False

    if len(optimizer.param_groups) == 0:
        print("❌ FAIL: The optimizer has no parameter groups.")
        return False

    for param_group in optimizer.param_groups:
        if abs(param_group["lr"] - lr) > 1e-12:
            print(
                "❌ FAIL: The optimizer does not use the learning rate "
                f"passed to initialize(). Got {param_group['lr']}, expected {lr}."
            )
            return False

        if abs(param_group["weight_decay"] - weight_decay) > 1e-12:
            print(
                "❌ FAIL: The optimizer does not use the weight decay "
                "passed to initialize(). "
                f"Got {param_group['weight_decay']}, expected {weight_decay}."
            )
            return False

    return True


def test_initialize():
    """
    Tests whether initialize() correctly creates the CNN and optimizer.
    """
    torch.manual_seed(0)

    lr = 0.123
    weight_decay = 0.045

    ################################################################
    # Part 1: Test normal functionality on CPU.
    ################################################################

    device = torch.device("cpu")

    try:
        model, optimizer = cnn.initialize(
            lr=lr,
            weight_decay=weight_decay,
            device=device,
        )
    except Exception as error:
        print(f"❌ FAIL: initialize() raised an error on CPU ({error}).")
        return

    if not isinstance(model, cnn.SimpleCNN):
        print(
            "❌ FAIL: initialize() should return a SimpleCNN instance "
            "as its first return value."
        )
        return

    if not isinstance(model, nn.Module):
        print(
            "❌ FAIL: The model returned by initialize() must inherit "
            "from torch.nn.Module."
        )
        return

    parameters = list(model.parameters())

    if len(parameters) == 0:
        print("❌ FAIL: The model returned by initialize() has no parameters.")
        return

    if not all(parameter.device == device for parameter in parameters):
        devices = sorted({str(parameter.device) for parameter in parameters})
        print(
            "❌ FAIL: Not all model parameters are on CPU during the CPU test. "
            f"Found parameters on {devices}."
        )
        return

    if not _check_optimizer_hyperparameters(optimizer, lr, weight_decay):
        return

    model_parameter_ids = {
        id(parameter)
        for parameter in model.parameters()
    }

    optimizer_parameter_ids = {
        id(parameter)
        for param_group in optimizer.param_groups
        for parameter in param_group["params"]
    }

    if optimizer_parameter_ids != model_parameter_ids:
        print(
            "❌ FAIL: The optimizer is not connected to exactly the parameters "
            "of the returned model."
        )
        return

    x = torch.randn(4, 3, 32, 32, device=device)

    try:
        with torch.no_grad():
            logits = model(x)
    except Exception as error:
        print(
            "❌ FAIL: The model returned by initialize() could not process "
            f"a CPU input tensor ({error})."
        )
        return

    if logits.shape != (4, 10):
        print(
            "❌ FAIL: The model returned by initialize() produced the wrong "
            f"output shape: got {logits.shape}, expected (4, 10)."
        )
        return

    ################################################################
    # Part 2: Test whether the requested device is actually respected.
    ################################################################

    test_device = torch.device("meta")

    try:
        device_model, device_optimizer = cnn.initialize(
            lr=lr,
            weight_decay=weight_decay,
            device=test_device,
        )
    except Exception as error:
        print(
            "❌ FAIL: initialize() raised an error when called with a "
            f"non-CPU device argument ({error}). "
            "Make sure the model is moved using model.to(device)."
        )
        return

    if not isinstance(device_model, cnn.SimpleCNN):
        print(
            "❌ FAIL: initialize() should return a SimpleCNN instance "
            "when called with the provided device argument."
        )
        return

    device_parameters = list(device_model.parameters())

    if len(device_parameters) == 0:
        print(
            "❌ FAIL: The model returned by initialize() has no parameters "
            "when called with the provided device argument."
        )
        return

    if not all(parameter.device == test_device for parameter in device_parameters):
        print(
            "❌ FAIL: initialize() does not seem to respect the device argument. "
            "Make sure the model is moved using model.to(device)."
        )
        return

    if not isinstance(optimizer, torch.optim.AdamW):
        print(
            "❌ FAIL: initialize() should return a torch.optim.AdamW optimizer "
            "as its second return value."
        )
        return

    print("✅ PASS: initialize() is correct.")


def test_training_step():
    """
    Tests whether training_step() performs one correct optimization step.
    """
    torch.manual_seed(0)

    ################################################################
    # Part 1: Check whether x and y are moved to the requested device.
    #
    # This uses a special internal PyTorch device so the test works
    # even on machines without CUDA.
    ################################################################

    class _DeviceCheckingModel(nn.Module):
        def __init__(self, expected_device):
            super().__init__()
            self.expected_device = expected_device
            self.seen_correct_device = False
            self.logits = nn.Parameter(torch.tensor([1.0, -1.0, 0.5]))

        def forward(self, x):
            self.seen_correct_device = x.device == self.expected_device

            if not self.seen_correct_device:
                raise RuntimeError("input tensor was not moved to the requested device")

            batch_size = x.shape[0]
            return self.logits.unsqueeze(0).expand(batch_size, -1)

    class _DeviceCheckingLoss(nn.Module):
        def __init__(self, expected_device):
            super().__init__()
            self.expected_device = expected_device
            self.seen_correct_device = False

        def forward(self, logits, y):
            self.seen_correct_device = y.device == self.expected_device

            if not self.seen_correct_device:
                raise RuntimeError("target tensor was not moved to the requested device")

            return logits.pow(2).mean()

    test_device = torch.device("meta")

    device_model = _DeviceCheckingModel(expected_device=test_device)
    device_loss_fn = _DeviceCheckingLoss(expected_device=test_device)
    device_optimizer = torch.optim.AdamW(device_model.parameters(), lr=0.1)

    x = torch.randn(4, 2)
    y = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    try:
        loss_value = cnn.training_step(
            cnn=device_model,
            optimizer=device_optimizer,
            loss_fn=device_loss_fn,
            x=x,
            y=y,
            device=test_device,
        )
    except Exception as error:
        print(
            "❌ FAIL: training_step() raised an error when called with a "
            f"non-CPU device argument ({error}). "
            "Make sure both x and y are moved using .to(device)."
        )
        return

    if not device_model.seen_correct_device:
        print(
            "❌ FAIL: training_step() does not seem to move the input tensor x "
            "to the requested device. Make sure to use x = x.to(device)."
        )
        return

    if not device_loss_fn.seen_correct_device:
        print(
            "❌ FAIL: training_step() does not seem to move the target tensor y "
            "to the requested device. Make sure to use y = y.to(device)."
        )
        return

    if not isinstance(loss_value, float):
        print(
            "❌ FAIL: training_step() should return the loss value as a "
            "Python float. Did you use loss.item()?"
        )
        return

    ################################################################
    # Part 2: Check the actual training behavior on CPU.
    ################################################################

    device = torch.device("cpu")

    model = nn.Linear(2, 3).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    x = torch.tensor(
        [
            [1.0, 2.0],
            [-1.0, 0.5],
            [0.3, -0.7],
            [2.0, -1.0],
        ],
        dtype=torch.float32,
    )

    y = torch.tensor([0, 2, 1, 0], dtype=torch.long)

    ################################################################
    # Compute the expected loss and expected SGD update.
    ################################################################

    initial_parameters = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    logits = model(x.to(device))
    expected_loss = loss_fn(logits, y.to(device))

    expected_gradients = torch.autograd.grad(
        expected_loss,
        list(model.parameters()),
    )

    expected_parameters_after_step = [
        parameter - optimizer.param_groups[0]["lr"] * gradient
        for parameter, gradient in zip(initial_parameters, expected_gradients)
    ]

    ################################################################
    # Add wrong old gradients.
    #
    # A correct training_step() must clear these before backpropagation.
    ################################################################

    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    try:
        loss_value = cnn.training_step(
            cnn=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            x=x,
            y=y,
            device=device,
        )
    except Exception as error:
        print(f"❌ FAIL: training_step() raised an error ({error}).")
        return

    ################################################################
    # Check returned loss value.
    ################################################################

    if not isinstance(loss_value, float):
        print(
            "❌ FAIL: training_step() should return the loss value as a "
            "Python float. Did you use loss.item()?"
        )
        return

    if abs(loss_value - expected_loss.item()) > 1e-6:
        print(
            "❌ FAIL: training_step() returned the wrong loss value. "
            f"Got {loss_value}, expected {expected_loss.item()}."
        )
        return

    ################################################################
    # Check that parameters changed.
    ################################################################

    current_parameters = list(model.parameters())

    for parameter_before, parameter_after in zip(
        initial_parameters,
        current_parameters,
    ):
        if torch.allclose(parameter_before, parameter_after):
            print("❌ FAIL: Model parameters did not change.")
            return

    ################################################################
    # Check that the update was exactly the expected SGD update.
    ################################################################

    for parameter, expected_parameter in zip(
        current_parameters,
        expected_parameters_after_step,
    ):
        if not torch.allclose(
            parameter.detach(),
            expected_parameter.detach(),
            atol=1e-6,
            rtol=1e-6,
        ):
            print("❌ FAIL: Model parameters were not updated correctly.")
            return

    print("✅ PASS: training_step() is correct.")
