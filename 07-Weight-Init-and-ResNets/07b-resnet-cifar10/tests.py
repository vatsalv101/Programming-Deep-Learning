import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import resnet


def _has_finite_gradients(module):
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    return (
        len(gradients) > 0
        and all(gradient is not None for gradient in gradients)
        and all(torch.isfinite(gradient).all() for gradient in gradients)
    )


def test_zero_pad_shortcut_module():
    """
    Tests the parameter-free shortcut used by CIFAR ResNet option A.
    """

    x = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8)

    try:
        shortcut = resnet.ZeroPadShortcut(out_channels=5, stride=2)
        out = shortcut(x)
    except Exception as error:
        print(f"FAIL: ZeroPadShortcut raised an error ({error}).")
        return False

    if not isinstance(shortcut, nn.Module):
        print("FAIL: ZeroPadShortcut must inherit from nn.Module.")
        return False

    has_maxpool = any(
        isinstance(module, nn.MaxPool2d)
        for module in shortcut.modules()
    )
    if not has_maxpool:
        print("FAIL: ZeroPadShortcut should use nn.MaxPool2d for spatial downsampling.")
        return False

    if out.shape != (2, 5, 4, 4):
        print(f"FAIL: ZeroPadShortcut returned shape {out.shape}, expected (2, 5, 4, 4).")
        return False

    expected_downsample = nn.MaxPool2d(kernel_size=1, stride=2)(x)
    if not torch.equal(out[:, :3], expected_downsample):
        print("FAIL: ZeroPadShortcut should preserve the downsampled input channels.")
        return False

    if not torch.equal(out[:, 3:], torch.zeros_like(out[:, 3:])):
        print("FAIL: ZeroPadShortcut should pad new channels with zeros.")
        return False

    if out.device != x.device or out.dtype != x.dtype:
        print("FAIL: ZeroPadShortcut should preserve the input device and dtype.")
        return False

    print("PASS: ZeroPadShortcut is correct.")
    return True


def test_projection_module():
    """
    Tests the learned 1x1 projection shortcut.
    """

    try:
        projection = resnet.ProjectionShortcut(3, 8, stride=2)
    except Exception as error:
        print(f"FAIL: ProjectionShortcut raised an error ({error}).")
        return False

    if not isinstance(projection, nn.Module):
        print("FAIL: ProjectionShortcut must inherit from nn.Module.")
        return False

    convs = [
        module
        for module in projection.modules()
        if isinstance(module, nn.Conv2d)
    ]

    if len(convs) == 0:
        print("FAIL: ProjectionShortcut should contain an nn.Conv2d layer.")
        return False

    conv1x1 = [
        conv
        for conv in convs
        if conv.kernel_size == (1, 1)
        and conv.stride == (2, 2)
        and conv.in_channels == 3
        and conv.out_channels == 8
    ]

    if len(conv1x1) == 0:
        print("FAIL: ProjectionShortcut should use a 1x1 Conv2d with the requested stride and channels.")
        return False

    x = torch.randn(4, 3, 32, 32)

    try:
        y = projection(x)
    except Exception as error:
        print(f"FAIL: projection shortcut could not process an input tensor ({error}).")
        return False

    if y.shape != (4, 8, 16, 16):
        print(f"FAIL: projection output has shape {y.shape}, expected (4, 8, 16, 16).")
        return False

    if sum(parameter.numel() for parameter in projection.parameters()) == 0:
        print("FAIL: projection shortcut should have trainable parameters.")
        return False

    print("PASS: ProjectionShortcut is correct.")
    return True


def test_basic_block():
    """
    Tests a residual block without fixing the exact internal module order.
    """

    torch.manual_seed(0)

    try:
        block = resnet.BasicBlock(
            in_channels=16,
            out_channels=32,
            stride=2,
            shortcut="projection",
        )
    except Exception as error:
        print(f"FAIL: BasicBlock could not be initialized ({error}).")
        return False

    if not isinstance(block, nn.Module):
        print("FAIL: BasicBlock must inherit from nn.Module.")
        return False

    has_conv = any(isinstance(module, nn.Conv2d) for module in block.modules())
    has_bn = any(isinstance(module, nn.BatchNorm2d) for module in block.modules())

    if not has_conv:
        print("FAIL: BasicBlock should contain convolutional layers.")
        return False

    if not has_bn:
        print("FAIL: BasicBlock should use nn.BatchNorm2d, as in the paper.")
        return False

    x = torch.randn(4, 16, 32, 32)

    try:
        y = block(x)
    except Exception as error:
        print(f"FAIL: BasicBlock.forward raised an error ({error}).")
        return False

    if y.shape != (4, 32, 16, 16):
        print(f"FAIL: BasicBlock output has shape {y.shape}, expected (4, 32, 16, 16).")
        return False

    if not torch.isfinite(y).all():
        print("FAIL: BasicBlock output contains non-finite values.")
        return False

    block.zero_grad()
    y.mean().backward()

    if not _has_finite_gradients(block):
        print("FAIL: Some BasicBlock parameters did not receive finite gradients.")
        return False

    print("PASS: BasicBlock is correct.")
    return True


def test_resnet_model():
    """
    Tests that ResNet behaves like a CIFAR-10 classifier.
    """

    torch.manual_seed(0)

    try:
        model = resnet.ResNet(
            num_blocks=(1, 1, 1),
            channels=(8, 16, 32),
            num_classes=10,
            shortcut="zero_pad",
        )
    except Exception as error:
        print(f"FAIL: ResNet could not be initialized ({error}).")
        return False

    if not isinstance(model, nn.Module):
        print("FAIL: ResNet must inherit from nn.Module.")
        return False

    if not any(isinstance(module, nn.Conv2d) for module in model.modules()):
        print("FAIL: ResNet should contain convolutional layers.")
        return False

    if not any(isinstance(module, nn.Linear) for module in model.modules()):
        print("FAIL: ResNet should contain a final nn.Linear classifier.")
        return False

    x = torch.randn(5, 3, 32, 32)

    try:
        logits = model(x)
    except Exception as error:
        print(f"FAIL: ResNet.forward raised an error ({error}).")
        return False

    if logits.shape != (5, 10):
        print(f"FAIL: ResNet.forward returned shape {logits.shape}, expected (5, 10).")
        return False

    if not torch.isfinite(logits).all():
        print("FAIL: ResNet.forward returned non-finite values.")
        return False

    if torch.all(logits >= 0):
        print("FAIL: ResNet should return raw logits, not ReLU outputs.")
        return False

    row_sums = logits.sum(dim=1)
    if torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3, rtol=1e-3):
        print("FAIL: ResNet should return raw logits, not softmax probabilities.")
        return False

    try:
        predicted = model.predict_class(x)
    except Exception as error:
        print(f"FAIL: ResNet.predict_class raised an error ({error}).")
        return False

    if predicted.shape != (5,) or predicted.dtype != torch.int64:
        print("FAIL: predict_class should return an int64 tensor of shape (batch_size,).")
        return False

    model.zero_grad()
    logits.mean().backward()

    if not _has_finite_gradients(model):
        print("FAIL: Some ResNet parameters did not receive finite gradients.")
        return False

    print("PASS: ResNet model is correct.")
    return True


def test_learning_rate_schedule():
    """
    Tests step learning-rate scheduling and the provided optimizer updater.
    """

    expected = {
        0: 0.1,
        1: 0.1,
        2: 0.01,
        3: 0.01,
        4: 0.001,
        5: 0.001,
    }

    for epoch, expected_lr in expected.items():
        try:
            lr = resnet.step_learning_rate(
                epoch=epoch,
                base_lr=0.1,
                milestones=(2, 4),
                gamma=0.1,
            )
        except Exception as error:
            print(f"FAIL: step_learning_rate raised an error ({error}).")
            return False

        if abs(lr - expected_lr) > 1e-12:
            print(f"FAIL: step_learning_rate({epoch}) returned {lr}, expected {expected_lr}.")
            return False

    model = nn.Linear(2, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    resnet.set_learning_rate(optimizer, 0.0123)

    for param_group in optimizer.param_groups:
        if abs(param_group["lr"] - 0.0123) > 1e-12:
            print("FAIL: set_learning_rate did not update all parameter groups.")
            return False

    print("PASS: learning-rate schedule is correct.")
    return True


def test_training_step_and_epoch():
    """
    Tests batch-level and epoch-level training behavior on a tiny dataset.
    """

    torch.manual_seed(0)
    device = torch.device("cpu")

    model = nn.Sequential(nn.Flatten(), nn.Linear(2, 3)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()

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

    initial_parameters = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    try:
        loss_value = resnet.training_step(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            x=x,
            y=y,
            device=device,
        )
    except Exception as error:
        print(f"FAIL: training_step raised an error ({error}).")
        return False

    if not isinstance(loss_value, float):
        print("FAIL: training_step should return a Python float.")
        return False

    if not all(
        not torch.allclose(before, after)
        for before, after in zip(initial_parameters, model.parameters())
    ):
        print("FAIL: training_step did not update all model parameters.")
        return False

    dataset = TensorDataset(
        torch.randn(8, 2),
        torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    try:
        mean_loss = resnet.train_one_epoch(
            model=model,
            train_loader=loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )
    except Exception as error:
        print(f"FAIL: train_one_epoch raised an error ({error}).")
        return False

    if not isinstance(mean_loss, float) or not torch.isfinite(torch.tensor(mean_loss)):
        print("FAIL: train_one_epoch should return a finite Python float.")
        return False

    print("PASS: training_step and train_one_epoch are correct.")
    return True


def test_evaluate_accuracy():
    """
    Tests accuracy evaluation without constraining the training loop.
    """

    class ConstantModel(nn.Module):
        def forward(self, x):
            logits = torch.zeros(x.shape[0], 3, device=x.device)
            logits[:, 1] = 1.0
            return logits

        def predict_class(self, x):
            return torch.argmax(self(x), dim=1)

    dataset = TensorDataset(
        torch.randn(5, 2),
        torch.tensor([1, 0, 1, 2, 1], dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    try:
        accuracy = resnet.evaluate_accuracy(
            model=ConstantModel(),
            data_loader=loader,
            device=torch.device("cpu"),
        )
    except Exception as error:
        print(f"FAIL: evaluate_accuracy raised an error ({error}).")
        return False

    if abs(accuracy - 0.6) > 1e-12:
        print(f"FAIL: evaluate_accuracy returned {accuracy}, expected 0.6.")
        return False

    print("PASS: evaluate_accuracy is correct.")
    return True


def run_all_tests():
    """
    Runs all tests in this file.
    """

    tests = [
        test_zero_pad_shortcut_module,
        test_projection_module,
        test_basic_block,
        test_resnet_model,
        test_learning_rate_schedule,
        test_training_step_and_epoch,
        test_evaluate_accuracy,
    ]

    results = [test() for test in tests]
    passed = sum(result is True for result in results)
    print(f"\nPassed {passed}/{len(tests)} tests.")
    return all(result is True for result in results)
