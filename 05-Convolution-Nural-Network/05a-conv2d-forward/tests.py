import ast
import inspect
import textwrap
import timeit

import torch
import torch.nn.functional as F

import conv2d
import visual


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


def _test_conv2d_forward(fn, expected_loops):
    fn_name = getattr(fn, "__name__", "function")

    loop_count = _count_python_loops(fn)
    if loop_count != expected_loops:
        print(
            f"❌ FAIL: {fn_name}() should contain exactly {expected_loops} Python loops, "
            f"but found {loop_count}."
        )
        return

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

            y_custom = fn(x, weight, bias, stride=cfg["stride"], padding=cfg["padding"])
            y_torch = F.conv2d(x, weight, bias, stride=cfg["stride"], padding=cfg["padding"])

            ok, _, msg = _allclose_with_error(y_custom, y_torch, atol=1e-8, rtol=1e-6)
            if not ok:
                print(f"❌ FAIL: {fn_name}() failed on test case {idx} ({msg}).")
                return

    print(f"✅ PASS: {fn_name}() is correct.")


def test_conv2d_forward_4_loops():
    """
    Tests conv2d_forward_4_loops against torch.nn.functional.conv2d.
    """
    _test_conv2d_forward(conv2d.conv2d_forward_4_loops, expected_loops=4)


def test_conv2d_forward_3_loops():
    """
    Tests conv2d_forward_3_loops against torch.nn.functional.conv2d.
    """
    _test_conv2d_forward(conv2d.conv2d_forward_3_loops, expected_loops=3)


def test_conv2d_forward_1_loop():
    """
    Tests conv2d_forward_1_loop against torch.nn.functional.conv2d.
    """
    _test_conv2d_forward(conv2d.conv2d_forward_1_loop, expected_loops=1)


def benchmark_conv2d_forward():
    """
    Benchmarks all Conv2D forward implementations and compares them to PyTorch.

    Returns:
        plotly.graph_objects.Figure: Benchmark plot.
    """
    print("Warning: runtime measurements can vary depending on hardware and current system load.")

    torch.manual_seed(0)

    batch_size = 8
    in_channels = 8
    out_channels = 16
    height, width = 32, 32
    kernel_size = 3
    stride = 1
    padding = 1

    x = torch.randn(batch_size, in_channels, height, width)
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
    bias = torch.randn(out_channels)

    num = 5
    repeat = 5

    times_4 = timeit.repeat(
        lambda: conv2d.conv2d_forward_4_loops(
            x, weight, bias, stride=stride, padding=padding
        ),
        number=num,
        repeat=repeat,
    )
    times_3 = timeit.repeat(
        lambda: conv2d.conv2d_forward_3_loops(
            x, weight, bias, stride=stride, padding=padding
        ),
        number=num,
        repeat=repeat,
    )
    times_1 = timeit.repeat(
        lambda: conv2d.conv2d_forward_1_loop(
            x, weight, bias, stride=stride, padding=padding
        ),
        number=num,
        repeat=repeat,
    )
    times_torch = timeit.repeat(
        lambda: F.conv2d(x, weight, bias, stride=stride, padding=padding),
        number=num,
        repeat=repeat,
    )

    return visual.plot_comparison(
        {
            "4 loops": times_4,
            "3 loops": times_3,
            "1 loop": times_1,
            "torch": times_torch,
        },
        title="Conv2D: Forward Pass",
        axis=["implementation", "time [s]"],
    )
