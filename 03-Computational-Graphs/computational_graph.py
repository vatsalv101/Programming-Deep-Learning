from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.patches import FancyArrowPatch

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

try:
    from IPython.display import SVG, Image, display
except Exception:  # pragma: no cover - optional in non-notebook use
    SVG = None
    Image = None
    display = None


@dataclass
class _Node:
    id: str
    name: str
    kind: str
    parents: List[str] = field(default_factory=list)
    forward: Any = None
    backward: Any = None
    order: int = 0


class CG:
    """
    Small notebook-friendly computational graph renderer.

    Example
    -------
    >>> g = CG()
    >>> g.input('x', id='x', forward=1, backward=16)
    >>> g.add_node('* 1/2', id='x_half', input_nodes=['x'], forward=0.5, backward=2)
    >>> g.render()
    """

    def __init__(
        self,
        *,
        precision: int = 2,
        dpi: int = 160,
        layer_spacing: float = 2.5,
        row_spacing: float = 1.6,
        node_height: float = 0.72,
        min_node_width: float = 0.72,
        font_size: int = 16,
        value_font_size: int = 12,
        edge_line_width: float = 1.1,
        value_column_gap: float = 0.16,
        route_grid_step: float = 0.16,
        value_vertical_gap: float = 0.12,
    ) -> None:
        self.precision = precision
        self.dpi = dpi
        self.layer_spacing = layer_spacing
        self.row_spacing = row_spacing
        self.node_height = node_height
        self.min_node_width = min_node_width
        self.font_size = font_size
        self.value_font_size = value_font_size
        self.edge_line_width = edge_line_width
        self.value_column_gap = value_column_gap
        self.route_grid_step = route_grid_step
        self.value_vertical_gap = value_vertical_gap

        self.nodes: Dict[str, _Node] = {}
        self.edges: List[Tuple[str, str]] = []
        self._order_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def input(
        self,
        name: str,
        *,
        id: Optional[str] = None,
        forward: Any = None,
        backward: Any = None,
    ) -> str:
        return self._add_node(
            name=name,
            id=id,
            kind="input",
            input_nodes=[],
            forward=forward,
            backward=backward,
        )

    def add_node(
        self,
        name: str,
        *,
        input_nodes: Sequence[str],
        id: Optional[str] = None,
        forward: Any = None,
        backward: Any = None,
    ) -> str:
        return self._add_node(
            name=name,
            id=id,
            kind="intermediate",
            input_nodes=list(input_nodes),
            forward=forward,
            backward=backward,
        )

    def output(
        self,
        name: str,
        *,
        input_nodes: Sequence[str],
        id: Optional[str] = None,
        forward: Any = None,
        backward: Any = None,
    ) -> str:
        return self._add_node(
            name=name,
            id=id,
            kind="output",
            input_nodes=list(input_nodes),
            forward=forward,
            backward=backward,
        )

    def render(
        self,
        *,
        filename: Optional[str | Path] = None,
        format: str = "svg",
        figsize: Optional[Tuple[float, float]] = None,
        show: bool = True,
        return_fig: bool = False,
        background: str = "white",
    ):
        if not self.nodes:
            raise ValueError("The graph is empty.")

        layers = self._compute_layers()
        node_widths = {node_id: self._node_width(node.name) for node_id, node in self.nodes.items()}
        positions = self._compute_positions(layers, node_widths)

        if figsize is None:
            max_rows = max(len(layer_nodes) for layer_nodes in layers.values())
            width = max(8.0, 2.8 + len(layers) * 2.4)
            height = max(3.5, 1.2 + max_rows * 1.45)
            figsize = (width, height)

        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        fig.patch.set_facecolor(background)
        ax.set_facecolor(background)

        # Edges first, so nodes sit on top.
        node_obstacles = self._build_node_obstacles(positions, node_widths)
        edge_cells_occupied: set[tuple[int, int]] = set()
        edge_segments_occupied: set[tuple[int, int, int, int]] = set()

        children_by_node: Dict[str, List[str]] = {node_id: [] for node_id in self.nodes}
        for src, dst in self.edges:
            children_by_node[src].append(dst)

        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        x_min = min(xs) - 1.4
        x_max = max(xs) + 1.8
        y_min = min(ys) - 1.6
        y_max = max(ys) + 1.6

        for src, dst in self.edges:
            x1, y1 = positions[src]
            x2, y2 = positions[dst]
            src_half = node_widths[src] / 2.0
            dst_half = node_widths[dst] / 2.0

            out_deg = max(1, len(children_by_node[src]))
            in_deg = max(1, len(self.nodes[dst].parents))
            out_idx = children_by_node[src].index(dst)
            in_idx = self.nodes[dst].parents.index(src)
            start_dy = (out_idx - (out_deg - 1) / 2.0) * 0.10
            end_dy = (in_idx - (in_deg - 1) / 2.0) * 0.10

            start = (x1 + src_half * 0.96, y1 + start_dy)
            end = (x2 - dst_half * 0.96, y2 + end_dy)

            route = self._route_edge_path(
                start=start,
                end=end,
                node_obstacles=node_obstacles,
                edge_cells_occupied=edge_cells_occupied,
                edge_segments_occupied=edge_segments_occupied,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
            )

            if len(route) <= 1:
                route = [start, end]

            rx = [p[0] for p in route]
            ry = [p[1] for p in route]
            ax.plot(
                rx,
                ry,
                color="black",
                lw=self.edge_line_width,
                zorder=1,
                solid_capstyle="round",
            )

            if len(route) >= 2:
                p0 = route[-2]
                p1 = route[-1]
                head = FancyArrowPatch(
                    p0,
                    p1,
                    arrowstyle="->",
                    mutation_scale=10,
                    linewidth=self.edge_line_width,
                    color="black",
                    zorder=1,
                    shrinkA=0,
                    shrinkB=0,
                )
                ax.add_patch(head)

        for node_id, node in sorted(self.nodes.items(), key=lambda item: item[1].order):
            x, y = positions[node_id]
            width = node_widths[node_id]
            patch = Ellipse(
                (x, y),
                width=width,
                height=self.node_height,
                facecolor="white",
                edgecolor="black",
                linewidth=1.0,
                zorder=2,
            )
            ax.add_patch(patch)
            ax.text(
                x,
                y,
                node.name,
                ha="center",
                va="center",
                fontsize=self.font_size,
                family="DejaVu Serif",
                zorder=3,
            )

            value_y_gap = self.node_height / 2.0 + self.value_vertical_gap
            fw_text = self._format_value(node.forward)
            bw_text = self._format_value(node.backward)

            # The axis is inverted so smaller y-values appear visually higher.
            if fw_text:
                ax.text(
                    x,
                    y - value_y_gap,
                    fw_text,
                    ha="center",
                    va="bottom",
                    fontsize=self.value_font_size,
                    family="DejaVu Sans Mono",
                    clip_on=False,
                    zorder=4,
                    bbox=dict(facecolor=background, edgecolor="none", pad=0.15),
                )
            if bw_text:
                ax.text(
                    x,
                    y + value_y_gap,
                    bw_text,
                    ha="center",
                    va="top",
                    fontsize=self.value_font_size,
                    family="DejaVu Sans Mono",
                    clip_on=False,
                    zorder=4,
                    bbox=dict(facecolor=background, edgecolor="none", pad=0.15),
                )

        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        ax.set_xlim(min(xs) - 1.2, max(xs) + 1.4)
        ax.set_ylim(min(ys) - 1.3, max(ys) + 1.3)
        ax.invert_yaxis()
        ax.axis("off")
        fig.tight_layout(pad=0.2)

        fmt = format.lower().replace(".", "")
        if fmt not in {"svg", "png", "pdf"}:
            raise ValueError("format must be one of: 'svg', 'png', 'pdf'.")

        if filename is not None:
            filename = Path(filename)
            if filename.suffix == "":
                filename = filename.with_suffix(f".{fmt}")
            fig.savefig(filename, bbox_inches="tight", pad_inches=0.2)

        if return_fig:
            return fig, ax

        if show:
            buf = BytesIO()
            fig.savefig(buf, format=fmt, bbox_inches="tight", pad_inches=0.2)
            buf.seek(0)
            if fmt == "svg" and SVG is not None and display is not None:
                display(SVG(data=buf.getvalue()))
            elif fmt == "png" and Image is not None and display is not None:
                display(Image(data=buf.getvalue()))
            else:
                plt.show()
        plt.close(fig)
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _add_node(
        self,
        *,
        name: str,
        id: Optional[str],
        kind: str,
        input_nodes: Sequence[str],
        forward: Any,
        backward: Any,
    ) -> str:
        node_id = str(id) if id is not None else self._make_id(name)
        if node_id in self.nodes:
            raise ValueError(f"Duplicate node id: {node_id!r}")

        missing = [p for p in input_nodes if p not in self.nodes]
        if missing:
            raise ValueError(
                f"Node {node_id!r} references missing parent nodes: {missing}"
            )

        self._order_counter += 1
        self.nodes[node_id] = _Node(
            id=node_id,
            name=name,
            kind=kind,
            parents=list(input_nodes),
            forward=forward,
            backward=backward,
            order=self._order_counter,
        )
        for parent in input_nodes:
            self.edges.append((parent, node_id))
        return node_id

    def _make_id(self, name: str) -> str:
        base = "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_") or "node"
        node_id = base
        i = 2
        while node_id in self.nodes:
            node_id = f"{base}_{i}"
            i += 1
        return node_id

    def _compute_layers(self) -> Dict[int, List[str]]:
        layer_of: Dict[str, int] = {}
        non_output_max = 0

        for node_id, node in sorted(self.nodes.items(), key=lambda item: item[1].order):
            if node.kind == "input":
                layer_of[node_id] = 0
            else:
                layer_of[node_id] = 1 + max(layer_of[parent] for parent in node.parents)
            if node.kind != "output":
                non_output_max = max(non_output_max, layer_of[node_id])

        # Force outputs onto the final column.
        for node_id, node in self.nodes.items():
            if node.kind == "output":
                layer_of[node_id] = non_output_max + 1

        layers: Dict[int, List[str]] = {}
        for node_id, layer in layer_of.items():
            layers.setdefault(layer, []).append(node_id)

        # Keep insertion order inside a layer.
        for layer in layers:
            layers[layer].sort(key=lambda node_id: self.nodes[node_id].order)
        return dict(sorted(layers.items()))

    def _compute_positions(
        self,
        layers: Dict[int, List[str]],
        node_widths: Dict[str, float],
    ) -> Dict[str, Tuple[float, float]]:
        positions: Dict[str, Tuple[float, float]] = {}
        effective_row_spacing = self._effective_row_spacing()

        # --- X positions (unchanged) ---
        x_by_layer: Dict[int, float] = {}
        ordered_layers = list(layers.keys())
        if ordered_layers:
            x_by_layer[ordered_layers[0]] = 0.0
        for prev_layer, next_layer in zip(ordered_layers, ordered_layers[1:]):
            right_extent = 0.0
            for node_id in layers[prev_layer]:
                node = self.nodes[node_id]
                text_span = self._value_text_width_units(node.forward, node.backward)
                right_extent = max(
                    right_extent,
                    node_widths[node_id] / 2.0 + self.value_column_gap + text_span,
                )
            extra_gap = max(0.0, right_extent - 1.25)
            x_by_layer[next_layer] = x_by_layer[prev_layer] + self.layer_spacing + extra_gap

        # --- Y positions ---
        # First layer: evenly spaced and centered (no parent info yet).
        # Later layers: each node's ideal y = mean of its already-placed parents' y,
        # then sort within layer by ideal y and spread to avoid overlaps.
        max_rows = max(len(node_ids) for node_ids in layers.values())
        total_height = (max_rows - 1) * effective_row_spacing

        for layer_idx in ordered_layers:
            node_ids = layers[layer_idx]
            x = x_by_layer[layer_idx]
            n = len(node_ids)

            if layer_idx == ordered_layers[0]:
                # First layer: simple centered even distribution.
                layer_height = (n - 1) * effective_row_spacing
                top = (total_height - layer_height) / 2.0
                for i, nid in enumerate(node_ids):
                    positions[nid] = (x, top + i * effective_row_spacing)
            else:
                # Ideal y = average y of already-placed parents.
                ideal: List[Optional[float]] = []
                for nid in node_ids:
                    parents_placed = [p for p in self.nodes[nid].parents if p in positions]
                    if parents_placed:
                        ideal.append(sum(positions[p][1] for p in parents_placed) / len(parents_placed))
                    else:
                        ideal.append(None)

                # Fill in ideals for nodes with no placed parents, centering
                # them around the mean of the known ideals.
                known = [y for y in ideal if y is not None]
                ref = sum(known) / len(known) if known else total_height / 2.0
                unknown_indices = [i for i in range(n) if ideal[i] is None]
                for k, i in enumerate(unknown_indices):
                    offset = k - (len(unknown_indices) - 1) / 2.0
                    ideal[i] = ref + offset * effective_row_spacing

                # Sort within the layer by ideal y to minimise edge crossings.
                order = sorted(range(n), key=lambda i: ideal[i])  # type: ignore[arg-type]
                sorted_ids = [node_ids[j] for j in order]
                sorted_ideal = [ideal[j] for j in order]  # type: ignore[index]

                # Spread: keep nodes as close to their ideals as possible
                # while enforcing the minimum gap.
                ys = self._spread_nodes(sorted_ideal, effective_row_spacing)  # type: ignore[arg-type]

                for nid, y in zip(sorted_ids, ys):
                    positions[nid] = (x, y)

        return positions

    def _spread_nodes(self, ideal_ys: List[float], min_spacing: float) -> List[float]:
        """Place nodes as close to *ideal_ys* (sorted ascending) as possible
        while keeping every consecutive gap >= *min_spacing*.

        Uses the block-merge algorithm which minimises the sum of squared
        deviations from the ideal positions.
        """
        n = len(ideal_ys)
        if n <= 1:
            return list(ideal_ys)

        # Each block owns a contiguous run of nodes.  Its center is the mean of
        # the ideal positions it contains; items within a block are spaced
        # exactly *min_spacing* apart.  When placing a new block would cause
        # overlap with the previous one, merge both into a single block.
        blocks: List[List[float]] = [[p] for p in ideal_ys]

        changed = True
        while changed:
            changed = False
            merged: List[List[float]] = []
            for blk in blocks:
                if merged:
                    prev = merged[-1]
                    prev_c = sum(prev) / len(prev)
                    curr_c = sum(blk) / len(blk)
                    prev_last = prev_c + (len(prev) - 1) / 2.0 * min_spacing
                    curr_first = curr_c - (len(blk) - 1) / 2.0 * min_spacing
                    if curr_first < prev_last + min_spacing - 1e-9:
                        merged[-1] = prev + blk
                        changed = True
                        continue
                merged.append(blk)
            blocks = merged

        result: List[float] = []
        for blk in blocks:
            center = sum(blk) / len(blk)
            start = center - (len(blk) - 1) / 2.0 * min_spacing
            result.extend(start + j * min_spacing for j in range(len(blk)))
        return result

    def _build_node_obstacles(
        self,
        positions: Dict[str, Tuple[float, float]],
        node_widths: Dict[str, float],
    ) -> List[Tuple[float, float, float, float]]:
        obstacles: List[Tuple[float, float, float, float]] = []
        for node_id, (x, y) in positions.items():
            half_w = node_widths[node_id] / 2.0 + 0.08
            half_h = self.node_height / 2.0 + 0.10
            obstacles.append((x - half_w, x + half_w, y - half_h, y + half_h))
        return obstacles

    def _point_in_obstacle(
        self,
        p: Tuple[float, float],
        obstacles: List[Tuple[float, float, float, float]],
    ) -> bool:
        x, y = p
        for x0, x1, y0, y1 in obstacles:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return True
        return False

    def _segment_key(self, a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int, int, int]:
        if a <= b:
            return (a[0], a[1], b[0], b[1])
        return (b[0], b[1], a[0], a[1])

    def _route_edge_path(
        self,
        *,
        start: Tuple[float, float],
        end: Tuple[float, float],
        node_obstacles: List[Tuple[float, float, float, float]],
        edge_cells_occupied: set[tuple[int, int]],
        edge_segments_occupied: set[tuple[int, int, int, int]],
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> List[Tuple[float, float]]:
        step = self.route_grid_step
        nx = max(4, int(round((x_max - x_min) / step)) + 1)
        ny = max(4, int(round((y_max - y_min) / step)) + 1)

        def to_idx(p: Tuple[float, float]) -> Tuple[int, int]:
            gx = int(round((p[0] - x_min) / step))
            gy = int(round((p[1] - y_min) / step))
            gx = min(max(gx, 0), nx - 1)
            gy = min(max(gy, 0), ny - 1)
            return gx, gy

        def to_point(idx: Tuple[int, int]) -> Tuple[float, float]:
            return (x_min + idx[0] * step, y_min + idx[1] * step)

        start_idx = to_idx(start)
        end_idx = to_idx(end)

        blocked: set[tuple[int, int]] = set()
        for ix in range(nx):
            for iy in range(ny):
                pt = to_point((ix, iy))
                if self._point_in_obstacle(pt, node_obstacles):
                    blocked.add((ix, iy))

        blocked.discard(start_idx)
        blocked.discard(end_idx)

        # Keep separation from previously routed edges to avoid intersections.
        for cell in edge_cells_occupied:
            if cell != start_idx and cell != end_idx:
                blocked.add(cell)

        moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        frontier: List[Tuple[float, Tuple[int, int]]] = []
        heapq.heappush(frontier, (0.0, start_idx))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        cost_so_far: Dict[Tuple[int, int], float] = {start_idx: 0.0}

        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        found = False
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == end_idx:
                found = True
                break

            for dx, dy in moves:
                nxt = (current[0] + dx, current[1] + dy)
                if not (0 <= nxt[0] < nx and 0 <= nxt[1] < ny):
                    continue
                if nxt in blocked:
                    continue

                seg_key = self._segment_key(current, nxt)
                if seg_key in edge_segments_occupied and nxt != end_idx:
                    continue

                new_cost = cost_so_far[current] + 1.0
                # Penalize bends mildly so paths look clean.
                prev = came_from.get(current)
                if prev is not None:
                    prev_dir = (current[0] - prev[0], current[1] - prev[1])
                    if prev_dir != (dx, dy):
                        new_cost += 0.15

                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + heuristic(nxt, end_idx)
                    heapq.heappush(frontier, (priority, nxt))
                    came_from[nxt] = current

        if not found:
            return [start, end]

        path_idx = [end_idx]
        cur = end_idx
        while cur != start_idx:
            cur = came_from[cur]
            path_idx.append(cur)
        path_idx.reverse()

        for i in range(len(path_idx) - 1):
            a = path_idx[i]
            b = path_idx[i + 1]
            edge_cells_occupied.add(a)
            edge_segments_occupied.add(self._segment_key(a, b))
        edge_cells_occupied.add(path_idx[-1])

        # Simplify collinear points.
        simplified: List[Tuple[int, int]] = []
        for idx in path_idx:
            if len(simplified) < 2:
                simplified.append(idx)
                continue
            a = simplified[-2]
            b = simplified[-1]
            c = idx
            if (b[0] - a[0], b[1] - a[1]) == (c[0] - b[0], c[1] - b[1]):
                simplified[-1] = c
            else:
                simplified.append(c)

        points = [to_point(idx) for idx in simplified]
        if points:
            points[0] = start
            points[-1] = end
        return points

    def _value_text_width_units(self, forward: Any, backward: Any) -> float:
        fw_text = self._format_value(forward)
        bw_text = self._format_value(backward)

        max_chars = 0
        for text in (fw_text, bw_text):
            if not text:
                continue
            max_chars = max(max_chars, max(len(line) for line in text.splitlines()))

        # Rough text-width estimate in data units used for inter-layer spacing.
        char_unit = 0.066 * (self.value_font_size / 12.0)
        return min(4.8, max_chars * char_unit)

    def _text_line_count(self, value: Any) -> int:
        text = self._format_value(value)
        if not text:
            return 0
        return len(text.splitlines())

    def _value_text_height_units(self, value: Any) -> float:
        line_count = self._text_line_count(value)
        if line_count == 0:
            return 0.0
        line_unit = 0.16 * (self.value_font_size / 12.0)
        return line_count * line_unit

    def _effective_row_spacing(self) -> float:
        if not self.nodes:
            return self.row_spacing

        max_fw_h = max(self._value_text_height_units(node.forward) for node in self.nodes.values())
        max_bw_h = max(self._value_text_height_units(node.backward) for node in self.nodes.values())
        required = self.node_height + 2.0 * self.value_vertical_gap + max_fw_h + max_bw_h + 0.26
        return max(self.row_spacing, required)

    def _node_width(self, label: str) -> float:
        clean = str(label)
        base = 0.42 + 0.12 * len(clean)
        if "\n" in clean:
            longest = max(len(line) for line in clean.splitlines())
            base = 0.42 + 0.12 * longest
        return max(self.min_node_width, min(base, 2.2))

    def _format_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if torch is not None and isinstance(value, torch.Tensor):
            if value.ndim == 0:
                value = value.item()
            else:
                value = value.detach().cpu().numpy()
        elif torch is not None and isinstance(value, torch.dtype):
            return str(value)
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            return self._format_number(float(value))

        arr = np.asarray(value)
        if arr.ndim == 0:
            return self._format_number(float(arr))

        # Determine the maximum decimal places needed across all entries so
        # that every element in the array is formatted with equal width.
        max_decimals = 0
        for v in arr.flatten():
            s = self._format_number(float(v))
            if "." in s:
                max_decimals = max(max_decimals, len(s.split(".")[1]))

        def format_uniform(x: float) -> str:
            if max_decimals == 0:
                return str(int(round(x)))
            return f"{x:.{max_decimals}f}"

        formatter = {
            "float_kind": lambda x: format_uniform(float(x)),
            "int_kind": lambda x: str(int(x)),
        }

        def format_1d(a1: np.ndarray) -> str:
            return np.array2string(
                a1,
                precision=self.precision,
                separator=" ",
                max_line_width=10_000,
                suppress_small=False,
                formatter=formatter,
            )

        if arr.ndim == 2 and arr.shape[0] == 1:
            return format_1d(arr[0])

        if arr.ndim == 2:
            # Keep one bracket pair per row and remove outer matrix wrapper.
            return "\n".join(format_1d(arr[i]) for i in range(arr.shape[0]))

        return np.array2string(
            arr,
            precision=self.precision,
            separator=" ",
            max_line_width=10_000,
            suppress_small=False,
            formatter=formatter,
        )

    def _format_number(self, x: float) -> str:
        if math.isclose(x, round(x), rel_tol=0.0, abs_tol=10 ** (-self.precision)):
            return str(int(round(x)))
        text = f"{x:.{self.precision}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        if text == "-0":
            text = "0"
        return text
