import ast
import inspect
import textwrap

import torch
import torch.nn as nn
import torch.nn.functional as F

import cnn
import layers


def _allclose_with_error(student, reference, atol=1e-8, rtol=1e-6):
    if student.shape != reference.shape:
        return False, None, f"wrong shape: got {student.shape}, expected {reference.shape}"

    max_abs_error = (student - reference).abs().max().item()

    if not torch.allclose(student, reference, atol=atol, rtol=rtol):
        return False, max_abs_error, f"max abs error: {max_abs_error:.6e}"

    return True, max_abs_error, None


def _count_python_loops(fn):
    source = inspect.getsource(fn)
    tree = ast.parse(textwrap.dedent(source))

    loop_nodes = (ast.For, ast.While, ast.AsyncFor)
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, loop_nodes):
            count += 1

    return count


def test_dropout_forward_backward():
    """
    Tests dropout in training mode, evaluation mode, and the backward pass.
    """
    if _count_python_loops(layers.dropout_forward_train) > 0:
        print(
            "FAIL: dropout_forward_train() should be vectorized and contain no Python loops."
        )
        return

    if _count_python_loops(layers.dropout_backward) > 0:
        print("FAIL: dropout_backward() should contain no Python loops.")
        return

    torch.manual_seed(0)

    x = torch.randn(2000, dtype=torch.float64) + 0.1
    p = 0.25

    out, cache = layers.dropout_forward_train(x, p=p)

    if out.shape != x.shape:
        print(f"FAIL: dropout output has wrong shape: got {out.shape}, expected {x.shape}.")
        return

    keep_prob = 1 - p
    nonzero = out != 0
    keep_fraction = nonzero.to(torch.float64).mean().item()

    if abs(keep_fraction - keep_prob) > 0.05:
        print(
            "FAIL: dropout kept the wrong fraction of activations: "
            f"got {keep_fraction:.3f}, expected about {keep_prob:.3f}."
        )
        return

    expected_kept_values = x[nonzero] / keep_prob
    ok, _, msg = _allclose_with_error(out[nonzero], expected_kept_values)
    if not ok:
        print(f"FAIL: kept activations are not scaled correctly ({msg}).")
        return

    dout = torch.randn_like(x)
    dx = layers.dropout_backward(dout, cache)
    mask, training = cache

    if not training:
        print("FAIL: training-mode dropout cache should mark training=True.")
        return

    ok, _, msg = _allclose_with_error(dx, dout * mask)
    if not ok:
        print(f"FAIL: dropout backward does not use the training mask ({msg}).")
        return

    out_eval, cache_eval = layers.dropout_forward_eval(x, p=p)
    ok, _, msg = _allclose_with_error(out_eval, x)
    if not ok:
        print(f"FAIL: evaluation-mode dropout should return the input unchanged ({msg}).")
        return

    dx_eval = layers.dropout_backward(dout, cache_eval)
    ok, _, msg = _allclose_with_error(dx_eval, dout)
    if not ok:
        print(f"FAIL: evaluation-mode dropout backward should pass gradients through ({msg}).")
        return

    print("PASS: dropout forward and backward are correct.")


def test_maxpool2d_forward():
    """
    Tests maxpool2d_forward against torch.nn.functional.max_pool2d.
    """
    loop_count = _count_python_loops(layers.maxpool2d_forward)
    if loop_count > 4:
        print(
            "FAIL: maxpool2d_forward() should use at most 4 Python loops, "
            f"but found {loop_count}."
        )
        return

    torch.manual_seed(1)

    test_cases = [
        dict(N=2, C=3, H=6, W=8, kernel_size=2, stride=2),
        dict(N=1, C=2, H=7, W=7, kernel_size=3, stride=1),
        dict(N=2, C=1, H=8, W=9, kernel_size=2, stride=3),
    ]

    for idx, cfg in enumerate(test_cases, start=1):
        x = torch.randn(
            cfg["N"], cfg["C"], cfg["H"], cfg["W"], dtype=torch.float32
        )

        out, cache = layers.maxpool2d_forward(
            x,
            kernel_size=cfg["kernel_size"],
            stride=cfg["stride"],
        )
        ref = F.max_pool2d(
            x,
            kernel_size=cfg["kernel_size"],
            stride=cfg["stride"],
        )

        ok, _, msg = _allclose_with_error(out, ref)
        if not ok:
            print(f"FAIL: maxpool2d_forward() failed on test case {idx} ({msg}).")
            return

        if len(cache) != 4:
            print(
                "WARNING: maxpool2d_forward() should return four quantities for the backward pass. Ignore this warning if you think your cache is correct."
            )
            #return

    print("PASS: maxpool2d_forward() is correct.")


def test_maxpool2d_backward():
    """
    Tests maxpool2d_backward against PyTorch autograd.
    """
    loop_count = _count_python_loops(layers.maxpool2d_backward)
    if loop_count > 4:
        print(
            "FAIL: maxpool2d_backward() should use at most 4 Python loops, "
            f"but found {loop_count}."
        )
        return

    torch.manual_seed(2)

    test_cases = [
        dict(N=2, C=2, H=6, W=6, kernel_size=2, stride=2),
        dict(N=1, C=3, H=5, W=7, kernel_size=3, stride=1),
    ]

    for idx, cfg in enumerate(test_cases, start=1):
        x = torch.randn(
            cfg["N"],
            cfg["C"],
            cfg["H"],
            cfg["W"],
            dtype=torch.float32,
            requires_grad=True,
        )

        out_ref = F.max_pool2d(
            x,
            kernel_size=cfg["kernel_size"],
            stride=cfg["stride"],
        )
        dout = torch.randn_like(out_ref)
        out_ref.backward(dout)
        dx_ref = x.grad.clone()

        x_plain = x.detach().clone()
        _, cache = layers.maxpool2d_forward(
            x_plain,
            kernel_size=cfg["kernel_size"],
            stride=cfg["stride"],
        )
        dx = layers.maxpool2d_backward(dout, cache)

        ok, _, msg = _allclose_with_error(dx, dx_ref, atol=1e-8, rtol=1e-6)
        if not ok:
            print(f"FAIL: maxpool2d_backward() failed on test case {idx} ({msg}).")
            return

    print("PASS: maxpool2d_backward() is correct.")