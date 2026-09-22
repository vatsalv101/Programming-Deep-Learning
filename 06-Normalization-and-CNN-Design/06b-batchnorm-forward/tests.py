import ast
import inspect
import textwrap

import torch
import torch.nn as nn

import batchnorm


def _allclose_with_error(student, reference, atol=1e-8, rtol=1e-6):
    if student.shape != reference.shape:
        return False, None, f"wrong shape: got {student.shape}, expected {reference.shape}"

    max_abs_error = (student - reference).abs().max().item()

    if not torch.allclose(student, reference, atol=atol, rtol=rtol):
        return False, max_abs_error, f"max abs error: {max_abs_error:.6e}"

    return True, max_abs_error, None


def _has_python_loops(fn):
    source = inspect.getsource(fn)
    tree = ast.parse(textwrap.dedent(source))

    loop_nodes = (ast.For, ast.While, ast.AsyncFor)
    for node in ast.walk(tree):
        if isinstance(node, loop_nodes):
            return True

    return False


def test_init_batchnorm2d():
    """
    Tests the initialization of the BatchNorm2d tensors.
    """
    gamma, beta, running_mean, running_var = batchnorm.init_batchnorm2d(4)

    if gamma.shape != (4,):
        print(f"❌ FAIL: gamma has wrong shape: got {gamma.shape}, expected (4,)")
        return

    if beta.shape != (4,):
        print(f"❌ FAIL: beta has wrong shape: got {beta.shape}, expected (4,)")
        return

    if running_mean.shape != (4,):
        print(
            "❌ FAIL: running_mean has wrong shape: "
            f"got {running_mean.shape}, expected (4,)"
        )
        return

    if running_var.shape != (4,):
        print(
            "❌ FAIL: running_var has wrong shape: "
            f"got {running_var.shape}, expected (4,)"
        )
        return

    if not torch.equal(gamma, torch.ones(4)):
        print("❌ FAIL: gamma should be initialized with ones.")
        return

    if not torch.equal(beta, torch.zeros(4)):
        print("❌ FAIL: beta should be initialized with zeros.")
        return

    if not torch.equal(running_mean, torch.zeros(4)):
        print("❌ FAIL: running_mean should be initialized with zeros.")
        return

    if not torch.equal(running_var, torch.ones(4)):
        print("❌ FAIL: running_var should be initialized with ones.")
        return

    print("✅ PASS: init_batchnorm2d() is correct.")


def test_batchnorm2d_forward_train():
    """
    Tests the training-mode forward pass, running statistics, and vectorization.
    """
    if _has_python_loops(batchnorm.batchnorm2d_forward_train):
        print(
            "❌ FAIL: batchnorm2d_forward_train contains a Python loop. "
            "The implementation should be vectorized."
        )
        return

    torch.manual_seed(0)

    gamma, beta, running_mean, running_var = batchnorm.init_batchnorm2d(3)
    gamma = torch.tensor([1.0, 0.5, 1.5], dtype=torch.float64)
    beta = torch.tensor([0.0, -1.0, 2.0], dtype=torch.float64)

    reference = nn.BatchNorm2d(
        num_features=3,
        eps=1e-5,
        momentum=0.2,
        affine=True,
        track_running_stats=True,
    ).double()

    with torch.no_grad():
        reference.weight.copy_(gamma)
        reference.bias.copy_(beta)
        reference.running_mean.copy_(running_mean)
        reference.running_var.copy_(running_var)

    x1 = torch.randn(4, 3, 5, 6, dtype=torch.float64) * 3 + 1
    x2 = torch.randn(2, 3, 4, 5, dtype=torch.float64) - 2

    y1_student, running_mean, running_var = batchnorm.batchnorm2d_forward_train(
        x1,
        gamma,
        beta,
        running_mean,
        running_var,
        eps=1e-5,
        momentum=0.2,
    )

    reference.train()
    y1_reference = reference(x1)

    ok, max_err, msg = _allclose_with_error(
        y1_student,
        y1_reference,
        atol=1e-8,
        rtol=1e-6,
    )
    if not ok:
        print(y1_student[0])
        print(y1_reference[0])
        print(f"❌ FAIL: training output after first batch does not match ({msg}).")
        return

    ok, max_err, msg = _allclose_with_error(
        running_mean,
        reference.running_mean,
        atol=1e-8,
        rtol=1e-6,
    )
    if not ok:
        print(f"❌ FAIL: running_mean after first batch does not match ({msg}).")
        return

    ok, max_err, msg = _allclose_with_error(
        running_var,
        reference.running_var,
        atol=1e-8,
        rtol=1e-6,
    )
    if not ok:
        print(f"❌ FAIL: running_var after first batch does not match ({msg}).")
        return

    y2_student, running_mean, running_var = batchnorm.batchnorm2d_forward_train(
        x2,
        gamma,
        beta,
        running_mean,
        running_var,
        eps=1e-5,
        momentum=0.2,
    )

    y2_reference = reference(x2)

    ok, max_err, msg = _allclose_with_error(
        y2_student,
        y2_reference,
        atol=1e-8,
        rtol=1e-6,
    )
    if not ok:
        print(f"❌ FAIL: training output after second batch does not match ({msg}).")
        return

    ok, max_err, msg = _allclose_with_error(
        running_mean,
        reference.running_mean,
        atol=1e-8,
        rtol=1e-6,
    )
    if not ok:
        print(f"❌ FAIL: running_mean after second batch does not match ({msg}).")
        return

    ok, max_err, msg = _allclose_with_error(
        running_var,
        reference.running_var,
        atol=1e-8,
        rtol=1e-6,
    )
    if not ok:
        print(f"❌ FAIL: running_var after second batch does not match ({msg}).")
        return

    print("✅ PASS: batchnorm2d_forward_train() is correct.")


def test_batchnorm2d_forward_eval():
    """
    Tests the evaluation-mode forward pass and vectorization.
    """
    if _has_python_loops(batchnorm.batchnorm2d_forward_eval):
        print(
            "❌ FAIL: batchnorm2d_forward_eval contains a Python loop. "
            "The implementation should be vectorized."
        )
        return

    torch.manual_seed(1)

    gamma = torch.tensor([1.2, 0.8, 1.5], dtype=torch.float64)
    beta = torch.tensor([-0.5, 0.25, 1.0], dtype=torch.float64)
    running_mean = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64)
    running_var = torch.tensor([1.5, 0.75, 2.5], dtype=torch.float64)

    x = torch.randn(2, 3, 4, 4, dtype=torch.float64) * 2 - 1

    y_student = batchnorm.batchnorm2d_forward_eval(
        x,
        gamma,
        beta,
        running_mean,
        running_var,
        eps=1e-5,
    ).double()

    reference = nn.BatchNorm2d(
        num_features=3,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ).double()

    with torch.no_grad():
        reference.weight.copy_(gamma)
        reference.bias.copy_(beta)
        reference.running_mean.copy_(running_mean)
        reference.running_var.copy_(running_var)

    reference.eval()
    y_reference = reference(x)

    ok, max_err, msg = _allclose_with_error(
        y_student,
        y_reference,
        atol=1e-8,
        rtol=1e-6,
    )
    if not ok:
        print(f"❌ FAIL: evaluation output does not match ({msg}).")
        return

    print("✅ PASS: batchnorm2d_forward_eval() is correct.")


def get_batchnorm_example_output():
    """
    Returns one small example input and output for visualization.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Input x and normalized output y.
    """
    torch.manual_seed(3)

    gamma, beta, running_mean, running_var = batchnorm.init_batchnorm2d(4)

    x = torch.randn(6, 4, 5, 5)
    x[:, 0] = 3.0 * x[:, 0] + 1.0
    x[:, 1] = 0.5 * x[:, 1] - 2.0
    x[:, 2] = 2.0 * x[:, 2] + 0.5
    x[:, 3] = 1.5 * x[:, 3] - 1.0

    y, _, _ = batchnorm.batchnorm2d_forward_train(
        x,
        gamma,
        beta,
        running_mean,
        running_var,
    )

    return x.detach(), y.detach()
