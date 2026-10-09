import html

import matplotlib


def is_notebook():
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        return "IPKernelApp" in shell.config
    except Exception:
        return False


if not is_notebook():
    matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt


def plot_training_loss(losses, ax=None):
    """
    Plots the language-model training loss.

    Args:
        losses (list): Cross-entropy loss values.
        ax (Axes): Optional matplotlib axes.

    Returns:
        ax (Axes): Matplotlib axes.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3))

    ax.plot(losses, color="#4c78a8")
    ax.set_title("Tiny LLM training loss")
    ax.set_xlabel("training step")
    ax.set_ylabel("cross entropy")
    ax.grid(alpha=0.25)
    return ax


def _moving_average(values, window):
    averaged = []
    for index in range(len(values)):
        start = max(0, index + 1 - window)
        chunk = values[start : index + 1]
        averaged.append(sum(chunk) / len(chunk))
    return averaged


def _format_seconds(seconds):
    if seconds is None:
        return "unknown"

    seconds = int(max(0, seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


class TrainingDashboard:
    """Live notebook dashboard for long language-model training runs."""

    def __init__(self, total_steps=None, smoothing=50):
        self.total_steps = total_steps
        self.smoothing = smoothing
        self.history = []
        self._display_handle = None

    def __call__(self, metrics):
        self.update(metrics)

    def update(self, metrics):
        self.history.append(dict(metrics))

        if not is_notebook():
            return metrics

        fig = self._build_figure(metrics)

        try:
            from IPython.display import display

            if self._display_handle is None:
                self._display_handle = display(fig, display_id=True)
            else:
                self._display_handle.update(fig)
        finally:
            plt.close(fig)

        return metrics

    def _build_figure(self, metrics):
        steps = [item["step"] for item in self.history]
        losses = [item["loss"] for item in self.history]
        recent_losses = [item.get("recent_loss", item["loss"]) for item in self.history]
        learning_rates = [item["learning_rate"] for item in self.history]
        speeds = [item["steps_per_second"] for item in self.history]

        fig = plt.figure(figsize=(12, 5), constrained_layout=True)
        grid = fig.add_gridspec(2, 3, width_ratios=[2.2, 1.7, 1.4])

        ax_loss = fig.add_subplot(grid[:, 0])
        ax_lr = fig.add_subplot(grid[0, 1])
        ax_speed = fig.add_subplot(grid[1, 1])
        ax_stats = fig.add_subplot(grid[:, 2])

        ax_loss.plot(steps, losses, color="#94a3b8", linewidth=1, alpha=0.65, label="loss")
        ax_loss.plot(steps, recent_losses, color="#2563eb", linewidth=2, label="recent avg")
        ax_loss.set_title("Training loss")
        ax_loss.set_xlabel("step")
        ax_loss.set_ylabel("cross entropy")
        ax_loss.grid(alpha=0.25)
        ax_loss.legend(loc="upper right")

        ax_lr.plot(steps, learning_rates, color="#9333ea", linewidth=2)
        ax_lr.set_title("Learning rate")
        ax_lr.set_xlabel("step")
        ax_lr.grid(alpha=0.25)

        ax_speed.plot(steps, speeds, color="#ea580c", linewidth=2)
        ax_speed.set_title("Speed")
        ax_speed.set_xlabel("step")
        ax_speed.set_ylabel("steps/s")
        ax_speed.grid(alpha=0.25)

        ax_stats.axis("off")
        ax_stats.text(
            0.0,
            1.0,
            self._stats_text(metrics),
            va="top",
            ha="left",
            family="monospace",
            fontsize=10,
        )

        return fig

    def _stats_text(self, metrics):
        total_steps = metrics.get("total_steps") or self.total_steps
        progress = ""
        if total_steps:
            progress = f" ({100.0 * metrics['step'] / total_steps:.1f}%)"

        epochs_done = metrics.get("epochs_done")
        epoch_text = "unknown" if epochs_done is None else f"{epochs_done:.2f}"
        checkpoint = metrics.get("checkpoint_path") or metrics.get("last_checkpoint_path") or "none yet"

        lines = [
            f"step:      {metrics['step']}/{total_steps}{progress}",
            f"loss:      {metrics['loss']:.4f}",
            f"avg loss:  {metrics.get('recent_loss', metrics['loss']):.4f}",
            f"lr:        {metrics['learning_rate']:.2e}",
            f"speed:     {metrics['steps_per_second']:.2f} steps/s",
            f"tokens/s:  {metrics.get('tokens_per_second', 0.0):.0f}",
            f"tokens:    {metrics.get('tokens_seen', 0)}",
            f"epochs:    {epoch_text}",
            f"elapsed:   {metrics.get('elapsed', _format_seconds(metrics.get('elapsed_seconds')))}",
            f"eta:       {metrics.get('eta', _format_seconds(metrics.get('eta_seconds')))}",
            f"runtime:   {metrics.get('estimated_total_runtime', _format_seconds(metrics.get('estimated_total_seconds')))}",
            "",
            "checkpoint:",
            checkpoint,
        ]
        return "\n".join(lines)


def plot_training_history(history):
    """
    Plots final loss, learning-rate, and speed curves from dashboard history.

    Args:
        history (list): Metrics dictionaries collected during training.

    Returns:
        fig (Figure): Matplotlib figure.
    """

    if len(history) == 0:
        raise ValueError("history is empty.")

    steps = [item["step"] for item in history]
    losses = [item["loss"] for item in history]
    learning_rates = [item["learning_rate"] for item in history]
    speeds = [item["steps_per_second"] for item in history]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), constrained_layout=True)

    axes[0].plot(steps, losses, color="#2563eb")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("step")
    axes[0].grid(alpha=0.25)

    axes[1].plot(steps, learning_rates, color="#9333ea")
    axes[1].set_title("Learning rate")
    axes[1].set_xlabel("step")
    axes[1].grid(alpha=0.25)

    axes[2].plot(steps, speeds, color="#ea580c")
    axes[2].set_title("Training speed")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("steps/s")
    axes[2].grid(alpha=0.25)

    return fig


def show_instruction_answer(instruction, input_text, answer):
    """
    Displays one instruction prompt and generated answer in a notebook.

    Args:
        instruction (str): Instruction text.
        input_text (str): Optional input text.
        answer (str): Generated model answer.

    Returns:
        html_text (HTML): HTML visualization in notebooks.
    """

    parts = [
        "<div style='font-family: ui-monospace, Consolas, monospace; "
        + "line-height:1.45; max-width:900px; color:#000000;'>",
        "<div style='margin-bottom:8px;'>",
        "<strong>Instruction</strong><br>",
        "<pre style='white-space:pre-wrap; margin:4px 0 0 0; padding:10px; "
        + "background:#f8fafc; border:1px solid #cbd5e1; color:#000000;'>",
        html.escape(instruction),
        "</pre></div>",
    ]

    if input_text:
        parts += [
            "<div style='margin-bottom:8px;'>",
            "<strong>Input</strong><br>",
            "<pre style='white-space:pre-wrap; margin:4px 0 0 0; padding:10px; "
            + "background:#f8fafc; border:1px solid #cbd5e1; color:#000000;'>",
            html.escape(input_text),
            "</pre></div>",
        ]

    parts += [
        "<div>",
        "<strong>Generated answer</strong><br>",
        "<pre style='white-space:pre-wrap; margin:4px 0 0 0; padding:10px; "
        + "background:#ecfdf5; border:1px solid #86efac; color:#000000;'>",
        html.escape(answer),
        "</pre></div></div>",
    ]

    html_text = "".join(parts)

    try:
        from IPython.display import HTML

        return HTML(html_text)
    except ImportError:
        return html_text
