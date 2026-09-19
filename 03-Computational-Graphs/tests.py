import importlib
import warnings
import torch
import cg
from computational_graph import CG

ATOL = 0.1


def _get_nodes_by_kind(g: CG, kind: str):
    return {nid: node for nid, node in g.nodes.items() if node.kind == kind}


def _values_close(actual, expected, label="", atol=ATOL):
    if actual is None:
        raise AssertionError(f"{label}: Value is None, expected {expected}")
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise AssertionError(
                f"{label}: Expected a tensor, got {type(actual).__name__}"
            )
        actual_f = actual.float()
        expected_f = expected.float()
        if actual_f.shape != expected_f.shape:
            raise AssertionError(
                f"{label}: Shape mismatch: got {actual_f.shape}, expected {expected_f.shape}"
            )
        if not torch.allclose(actual_f, expected_f, atol=atol):
            raise AssertionError(
                f"{label}: Values mismatch:\n  got:      {actual}\n  expected: {expected}"
            )
    else:
        if isinstance(actual, torch.Tensor):
            if actual.numel() != 1:
                raise AssertionError(
                    f"{label}: Expected scalar, got tensor with shape {actual.shape}"
                )
            actual = actual.item()
        if abs(float(actual) - float(expected)) > atol:
            raise AssertionError(
                f"{label}: Value mismatch: got {actual}, expected {expected}"
            )


def _check_shape(actual, expected, label=""):
    if actual is None:
        raise AssertionError(f"{label}: Value is None, expected {expected}")
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise AssertionError(
                f"{label}: Expected a tensor, got {type(actual).__name__}"
            )
        if actual.shape != expected.shape:
            raise AssertionError(
                f"{label}: Shape mismatch: got {actual.shape}, expected {expected.shape}"
            )
    else:
        if isinstance(actual, torch.Tensor) and actual.numel() > 1:
            raise AssertionError(
                f"{label}: Expected scalar, got tensor with shape {actual.shape}"
            )


def _test_graph(graph_fn, expected_inputs, min_intermediates, expected_output_fwd, expected_output_bwd):
    """
    Generic test for a computational graph.

    Parameters
    ----------
    graph_fn : callable
        Function that returns a CG instance (e.g. cg.graph_1).
    expected_inputs : dict
        Mapping from node id to {"forward": ..., "backward": ...}.
    min_intermediates : int
        Minimum number of intermediate nodes expected.
    expected_output_fwd : value
        Expected forward value of the output node.
    expected_output_bwd : value
        Expected backward value of the output node.
    """
    importlib.reload(cg)
    g = graph_fn()
    passed = 0
    failed = 0

    input_nodes = _get_nodes_by_kind(g, "input")
    intermediate_nodes = _get_nodes_by_kind(g, "intermediate")
    output_nodes = _get_nodes_by_kind(g, "output")

    # --- Input nodes: correct count ---
    try:
        assert len(input_nodes) == len(expected_inputs), (
            f"Expected {len(expected_inputs)} input node(s), got {len(input_nodes)}. "
            f"Expected ids: {list(expected_inputs.keys())}, got: {list(input_nodes.keys())}"
        )
        passed += 1
        print(f"  ✓ Correct number of input nodes ({len(expected_inputs)})")
    except AssertionError as e:
        failed += 1
        print(f"  ✗ {e}")

    # --- Input nodes: correct ids, shapes, values ---
    for node_id, expected in expected_inputs.items():
        label = f"Input node '{node_id}'"
        try:
            assert node_id in input_nodes, (
                f"{label}: Not found. Existing input node ids: {list(input_nodes.keys())}"
            )
            node = input_nodes[node_id]

            _check_shape(node.forward, expected["forward"], label=f"{label} forward shape")
            _check_shape(node.backward, expected["backward"], label=f"{label} backward shape")
            _values_close(node.forward, expected["forward"], label=f"{label} forward value")
            _values_close(node.backward, expected["backward"], label=f"{label} backward value")

            passed += 1
            print(f"  ✓ {label}: correct shape and values")
        except AssertionError as e:
            failed += 1
            print(f"  ✗ {e}")

    # --- Intermediate nodes: minimum count (warning only) ---
    if len(intermediate_nodes) < min_intermediates:
        warnings.warn(
            f"  ⚠ Expected at least {min_intermediates} intermediate node(s), "
            f"got {len(intermediate_nodes)}. Your graph might be missing steps."
        )
        print(
            f"  ⚠ WARNING: Expected at least {min_intermediates} intermediate node(s), "
            f"got {len(intermediate_nodes)}. Your graph might be missing steps."
        )
    else:
        passed += 1
        print(f"  ✓ Sufficient intermediate nodes ({len(intermediate_nodes)} >= {min_intermediates})")

    # --- Output node: correct count ---
    try:
        assert len(output_nodes) == 1, (
            f"Expected exactly 1 output node, got {len(output_nodes)}: {list(output_nodes.keys())}"
        )
        passed += 1
        print(f"  ✓ Correct number of output nodes (1)")
    except AssertionError as e:
        failed += 1
        print(f"  ✗ {e}")

    # --- Output node: shape and values ---
    if len(output_nodes) == 1:
        out_id, out_node = next(iter(output_nodes.items()))
        label = f"Output node '{out_id}'"
        try:
            _check_shape(out_node.forward, expected_output_fwd, label=f"{label} forward shape")
            _check_shape(out_node.backward, expected_output_bwd, label=f"{label} backward shape")
            _values_close(out_node.forward, expected_output_fwd, label=f"{label} forward value")
            _values_close(out_node.backward, expected_output_bwd, label=f"{label} backward value")
            passed += 1
            print(f"  ✓ {label}: correct shape and values")
        except AssertionError as e:
            failed += 1
            print(f"  ✗ {e}")

    # --- Summary ---
    total = passed + failed
    if failed == 0:
        print(f"\n  All {passed} checks passed! ✓")
    else:
        print(f"\n  {passed}/{total} checks passed, {failed} failed ✗")


# ======================================================================
# Graph 1
# ======================================================================
def test_graph_1():
    print("Testing Graph 1:")
    _test_graph(
        graph_fn=cg.graph_1,
        expected_inputs={
            "x": {"forward": 1, "backward": 16},
            "y": {"forward": 5, "backward": 3},
            "z": {"forward": 2, "backward": -0.42},
        },
        min_intermediates=6,
        expected_output_fwd=15.22,
        expected_output_bwd=1,
    )


# ======================================================================
# Graph 2
# ======================================================================
def test_graph_2():
    print("Testing Graph 2:")
    _test_graph(
        graph_fn=cg.graph_2,
        expected_inputs={
            "x": {"forward": 2, "backward": 0},
            "y": {"forward": 3, "backward": 0},
            "z": {"forward": 4, "backward": 0.76},
        },
        min_intermediates=4,
        expected_output_fwd=-0.65,
        expected_output_bwd=1,
    )


# ======================================================================
# Graph 3
# ======================================================================
def test_graph_3():
    print("Testing Graph 3:")
    _test_graph(
        graph_fn=cg.graph_3,
        expected_inputs={
            "W": {
                "forward": torch.tensor([[0.10, 0.05], [0.25, 0.15], [0.10, 0.40]]),
                "backward": torch.tensor([[0.25, 0.50], [0.11, 0.22], [0.04, 0.08]]),
            },
            "x": {
                "forward": torch.tensor([[1.0], [2.0]]),
                "backward": torch.tensor([[0.06], [0.05]]),
            },
            "b": {
                "forward": torch.tensor([[0.10], [0.15], [0.20]]),
                "backward": torch.tensor([[0.25], [0.11], [0.04]]),
            },
            "V": {
                "forward": torch.tensor([[1.00, 0.50, 0.20]]),
                "backward": torch.tensor([[0.57, 0.67, 0.75]]),
            },
        },
        min_intermediates=3,
        expected_output_fwd=1.06,
        expected_output_bwd=1,
    )


# ======================================================================
# Graph 4
# ======================================================================
def test_graph_4():
    print("Testing Graph 4:")
    _test_graph(
        graph_fn=cg.graph_4,
        expected_inputs={
            "x": {
                "forward": torch.tensor([[1.0], [2.0]]),
                "backward": torch.tensor([[-0.45], [-0.18]]),
            },
            "W": {
                "forward": torch.tensor([[0.5, 0.25], [0.2, -0.4]]),
                "backward": torch.tensor([[-0.9, 0.0], [-1.8, 0.0]]),
            },
            "V": {
                "forward": torch.tensor([[3.0], [2.0]]),
                "backward": torch.tensor([[-0.27], [0.0]]),
            },
            "y": {
                "forward": 3,
                "backward": 0.3,
            },
        },
        min_intermediates=6,
        expected_output_fwd=0.045,
        expected_output_bwd=1,
    )
