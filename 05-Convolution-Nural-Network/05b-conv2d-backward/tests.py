import ast
import inspect
import textwrap

import torch
import torch.nn.functional as F

import conv2d


def _allclose_with_error(student, reference, atol=1e-6, rtol=1e-5):
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


def _scalar_conv_loss(x, weight, bias, dout, stride, padding):
    out, _ = conv2d.conv2d_forward_4_loops(
        x, weight, bias, stride=stride, padding=padding
    )
    return torch.sum(out * dout)


def _numerical_gradient(tensor, loss_fn, eps=1e-6):
    grad = torch.zeros_like(tensor)
    flat_tensor = tensor.view(-1)
    flat_grad = grad.view(-1)

    for idx in range(flat_tensor.numel()):
        original = flat_tensor[idx].item()

        flat_tensor[idx] = original + eps
        loss_pos = loss_fn().item()

        flat_tensor[idx] = original - eps
        loss_neg = loss_fn().item()

        flat_tensor[idx] = original
        flat_grad[idx] = (loss_pos - loss_neg) / (2 * eps)

    return grad


def test_conv2d_forward():
    """
    Tests the provided Conv2D forward implementation against
    torch.nn.functional.conv2d.
    """

    for seed in range(10):

        torch.manual_seed(seed)

        test_cases = [
            dict(N=2, C_in=1, H=5, W=6, C_out=2, K_h=3, K_w=3, stride=1, padding=0),
            dict(N=1, C_in=3, H=8, W=7, C_out=4, K_h=3, K_w=3, stride=2, padding=1),
            dict(N=2, C_in=2, H=9, W=10, C_out=3, K_h=2, K_w=4, stride=1, padding=2),
        ]

        for idx, cfg in enumerate(test_cases, start=1):
            x = torch.randn(cfg["N"], cfg["C_in"], cfg["H"], cfg["W"], dtype=torch.float64)
            weight = torch.randn(
                cfg["C_out"], cfg["C_in"], cfg["K_h"], cfg["K_w"], dtype=torch.float64
            )
            bias = torch.randn(cfg["C_out"], dtype=torch.float64)

            out, cache = conv2d.conv2d_forward_4_loops(
                x, weight, bias, stride=cfg["stride"], padding=cfg["padding"]
            )
            ref = F.conv2d(x, weight, bias, stride=cfg["stride"], padding=cfg["padding"])

            ok, _, msg = _allclose_with_error(out, ref, atol=1e-8, rtol=1e-6)
            if not ok:
                print(f"❌ FAIL: conv2d_forward_4_loops() failed on test case {idx} ({msg}).")
                return

            if len(cache) != 5:
                print(
                    "❌ FAIL: conv2d_forward_4_loops() should return "
                    "cache = (x, weight, bias, stride, padding)."
                )
                return

    print("✅ PASS: conv2d_forward_4_loops() is correct.")


def test_conv2d_backward():
    """
    Tests the full backward pass against PyTorch autograd, numerical gradients,
    and the intended loop structure.
    """
    loop_count = _count_python_loops(conv2d.conv2d_backward)

    if loop_count > 4:
        print(
            "❌ FAIL: conv2d_backward() should use at most 4 Python loops, "
            f"but found {loop_count}."
        )
        return

    for seed in range(10):
        torch.manual_seed(seed)

        autograd_test_cases = [
            dict(N=2, C_in=2, H=5, W=6, C_out=3, K_h=3, K_w=2, stride=1, padding=1),
            dict(N=1, C_in=3, H=6, W=5, C_out=2, K_h=2, K_w=3, stride=2, padding=1),
        ]

        for idx, cfg in enumerate(autograd_test_cases, start=1):
            x = torch.randn(
                cfg["N"],
                cfg["C_in"],
                cfg["H"],
                cfg["W"],
                dtype=torch.float64,
                requires_grad=True,
            )
            weight = torch.randn(
                cfg["C_out"],
                cfg["C_in"],
                cfg["K_h"],
                cfg["K_w"],
                dtype=torch.float64,
                requires_grad=True,
            )
            bias = torch.randn(cfg["C_out"], dtype=torch.float64, requires_grad=True)

            out_ref = F.conv2d(
                x,
                weight,
                bias,
                stride=cfg["stride"],
                padding=cfg["padding"],
            )
            dout = torch.randn_like(out_ref)
            out_ref.backward(dout)

            dx_ref = x.grad.clone()
            dweight_ref = weight.grad.clone()
            db_ref = bias.grad.clone()

            x_plain = x.detach().clone()
            weight_plain = weight.detach().clone()
            bias_plain = bias.detach().clone()

            _, cache = conv2d.conv2d_forward_4_loops(
                x_plain,
                weight_plain,
                bias_plain,
                stride=cfg["stride"],
                padding=cfg["padding"],
            )
            dx, dweight, db = conv2d.conv2d_backward(dout, cache)

            ok, _, msg = _allclose_with_error(dx, dx_ref, atol=1e-8, rtol=1e-6)
            if not ok:
                print(
                    f"❌ FAIL: dx does not match PyTorch autograd on test case {idx} ({msg})."
                )
                return

            ok, _, msg = _allclose_with_error(dweight, dweight_ref, atol=1e-8, rtol=1e-6)
            if not ok:
                print(
                    f"❌ FAIL: dweight does not match PyTorch autograd on test case {idx} ({msg})."
                )
                return

            ok, _, msg = _allclose_with_error(db, db_ref, atol=1e-8, rtol=1e-6)
            if not ok:
                print(
                    f"❌ FAIL: db does not match PyTorch autograd on test case {idx} ({msg})."
                )
                return


        x = torch.randn(1, 2, 4, 4, dtype=torch.float64)
        weight = torch.randn(2, 2, 3, 2, dtype=torch.float64)
        bias = torch.randn(2, dtype=torch.float64)
        stride = 2
        padding = 1

        out, cache = conv2d.conv2d_forward_4_loops(
            x, weight, bias, stride=stride, padding=padding
        )
        dout = torch.randn_like(out)

        dx, dweight, db = conv2d.conv2d_backward(dout, cache)

        def loss_fn():
            return _scalar_conv_loss(x, weight, bias, dout, stride, padding)

        dx_num = _numerical_gradient(x, loss_fn)
        dweight_num = _numerical_gradient(weight, loss_fn)
        db_num = _numerical_gradient(bias, loss_fn)

        ok, _, msg = _allclose_with_error(dx, dx_num, atol=1e-8, rtol=1e-5)
        if not ok:
            print(f"❌ FAIL: dx does not match the numerical gradient ({msg}).")
            return

        ok, _, msg = _allclose_with_error(dweight, dweight_num, atol=1e-8, rtol=1e-5)
        if not ok:
            print(f"❌ FAIL: dweight does not match the numerical gradient ({msg}).")
            return

        ok, _, msg = _allclose_with_error(db, db_num, atol=1e-8, rtol=1e-5)
        if not ok:
            print(f"❌ FAIL: db does not match the numerical gradient ({msg}).")
            return

    print("✅ PASS: conv2d_backward() is correct.")
