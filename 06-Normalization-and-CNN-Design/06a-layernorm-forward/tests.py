import ast
import inspect
import textwrap

import torch
import torch.nn as nn

import layernorm


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


def test_init_layernorm1d():
    """
    Tests the initialization of the LayerNorm1d tensors.
    """
    gamma, beta = layernorm.init_layernorm1d(4)

    if gamma.shape != (4,):
        print(f"❌ FAIL: gamma has wrong shape: got {gamma.shape}, expected (4,)")
        return

    if beta.shape != (4,):
        print(f"❌ FAIL: beta has wrong shape: got {beta.shape}, expected (4,)")
        return

    if not torch.equal(gamma, torch.ones(4)):
        print("❌ FAIL: gamma should be initialized with ones.")
        return

    if not torch.equal(beta, torch.zeros(4)):
        print("❌ FAIL: beta should be initialized with zeros.")
        return

    print("✅ PASS: init_layernorm1d() is correct.")


def test_layernorm1d_forward():
    """
    Tests the forward pass of the LayerNorm1d layer.
    """
    if _has_python_loops(layernorm.layernorm1d_forward):
        print(
            "❌ FAIL: layernorm1d_forward contains a Python loop. "
            "The implementation should be vectorized."
        )
        return

    torch.manual_seed(0)

    gamma, beta = layernorm.init_layernorm1d(3)
    gamma = torch.tensor([1.0, 0.5, 1.5], dtype=torch.float64)
    beta = torch.tensor([0.0, -1.0, 2.0], dtype=torch.float64)

    reference = nn.LayerNorm(
        normalized_shape=3,
        eps=1e-5,
        elementwise_affine=True,
    ).double()

    with torch.no_grad():
        reference.weight.copy_(gamma)
        reference.bias.copy_(beta)

    x1 = torch.randn(4, 3, dtype=torch.float64) * 3 + 1
    x2 = torch.randn(2, 3, dtype=torch.float64) - 2

    y1_student = layernorm.layernorm1d_forward(
        x1,
        gamma,
        beta,
        eps=1e-5,
    )

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

    y2_student = layernorm.layernorm1d_forward(
        x2,
        gamma,
        beta,
        eps=1e-5,
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

    print("✅ PASS: layernorm1d_forward() is correct.")


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
