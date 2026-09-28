import math

import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from PIL import Image

import data


def _denormalize(images):
    mean = torch.tensor(data.AFHQ_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(data.AFHQ_STD).view(1, 3, 1, 1)
    return images * std + mean


def _image_grid(
    images,
    title,
    cols=4,
    top_margin=80,
    vertical_spacing=0.04,
):
    n = images.shape[0]
    cols = min(cols, n)
    rows = math.ceil(n / cols)

    images = _denormalize(images)
    images = images.clamp(0.0, 1.0).detach().cpu()
    images = (255 * images.permute(0, 2, 3, 1)).to(torch.uint8).numpy()

    fig = make_subplots(
        rows=rows,
        cols=cols,
        horizontal_spacing=0.03,
        vertical_spacing=vertical_spacing,
    )

    for index in range(n):
        row = index // cols + 1
        col = index % cols + 1
        fig.add_trace(go.Image(z=images[index]), row=row, col=col)

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        title=dict(text=title, y=0.98),
        showlegend=False,
        height=160 * rows + top_margin,
        width=180 * cols,
        margin=dict(l=20, r=20, t=top_margin, b=20),
    )

    return fig


def show_training_stats(history, title="GAN Training"):
    """
    Plots training and validation metrics over epochs.

    Expected history keys:
        train_loss, train_gen_loss, train_disc_loss
        val_loss, val_gen_loss, val_disc_loss
    """

    metrics = [
        ("gen_loss", "Gen. Loss"),
        ("disc_loss", "Disc. Loss"),
        ("disc_loss_real", "Disc. Loss (Real)"),
        ("disc_loss_fake", "Disc. Loss (Fake)"),
    ]

    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=["Training Gen. Loss",
            "Training Disc. Loss",
            "Training Disc. Loss (Real)",
            "Training Disc. Loss (Fake)",
            "Validation Gen. Loss",
            "Validation Disc. Loss",
            "Validation Disc. Loss (Real)",
            "Validation Disc. Loss (Fake)",
        ],
    )

    for col, (key, label) in enumerate(metrics, start=1):
        train_key = f"train_{key}"
        val_key = f"val_{key}"

        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(history[train_key]) + 1)),
                y=history[train_key],
                mode="lines+markers",
                name=train_key,
            ),
            row=1,
            col=col,
        )

        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(history[val_key]) + 1)),
                y=history[val_key],
                mode="lines+markers",
                name=val_key,
            ),
            row=2,
            col=col,
        )

        fig.update_yaxes(title_text=label, type="log", row=1, col=col)
        fig.update_yaxes(title_text=label, type="log", row=2, col=col)

    fig.update_xaxes(title_text="Epoch")
    fig.update_layout(
        title=title,
        showlegend=False,
        height=600,
        width=1300,
    )

    return fig

def animate_interpolation(samples, title="GAN Interpolation", savepath="./interpolation.gif", fps=8):
    """
    Generate an animated Plotly figure and optionally save interpolated samples as a gif.

    Args:
        samples (torch.Tensor): Interpolated samples of shape (steps, 3, H, W).
        title (str): Title for the plot.
        savepath (str | None): Path to save the gif. If None, no gif is saved.
        fps (int): Frames per second for the gif.

    Returns:
        fig (Figure): Plotly animated figure.
    """
    if samples.ndim != 4:
        raise ValueError(f"Expected samples with shape (steps, 3, H, W), got {samples.shape}")

    samples = _denormalize(samples).clamp(0.0, 1.0).detach().cpu()
    images = (255 * samples.permute(0, 2, 3, 1)).to(torch.uint8).numpy()

    if savepath is not None:
        pil_images = [Image.fromarray(img) for img in images]
        pil_images[0].save(savepath, save_all=True, append_images=pil_images[1:], duration=int(1000 / fps), loop=0)

    fig = go.Figure(
        data=[go.Image(z=images[0])],
        frames=[go.Frame(data=[go.Image(z=images[i])], name=str(i)) for i in range(len(images))],
    )

    fig.update_layout(
        title=title,
        width=400,
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        showlegend=False,
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(
                label="Play",
                method="animate",
                args=[None, dict(frame=dict(duration=int(1000 / fps), redraw=True), fromcurrent=True, transition=dict(duration=0))],
            )],
        )],
        sliders=[dict(steps=[
            dict(
                method="animate",
                label=str(i),
                args=[[str(i)], dict(frame=dict(duration=0, redraw=True), mode="immediate", transition=dict(duration=0))],
            )
            for i in range(len(images))
        ])],
    )

    return fig