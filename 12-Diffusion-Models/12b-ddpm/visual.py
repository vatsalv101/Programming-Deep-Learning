import math
import numpy as np
import plotly.graph_objects as go
import torch
from torchvision.utils import make_grid
from plotly.subplots import make_subplots
from PIL import Image

import data


def _denormalize(images):
    mean = torch.tensor(
        data.MEAN,
        device=images.device,
        dtype=images.dtype,
    ).view(1, -1, 1, 1)

    std = torch.tensor(
        data.STD,
        device=images.device,
        dtype=images.dtype,
    ).view(1, -1, 1, 1)

    return images * std + mean


def _to_uint8_rgb(images):
    """
    Converts normalized image batch to uint8 RGB for Plotly/PIL.

    Input:  (B, C, H, W), C can be 1 or 3
    Output: (B, H, W, 3)
    """
    images = _denormalize(images)
    images = images.clamp(0.0, 1.0).detach().cpu()

    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)

    images = images.permute(0, 2, 3, 1)
    images = (255 * images).clamp(0, 255).to(torch.uint8).numpy()

    return images


def _image_grid(images, title, cols=4, top_margin=80, vertical_spacing=0.04):
    n = images.shape[0]
    cols = min(cols, n)
    rows = math.ceil(n / cols)

    images = _to_uint8_rgb(images)

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


def show_images(images, title="Images", cols=4):
    return _image_grid(images, title=title, cols=cols)


def show_training_stats(history, title="Diffusion Training"):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(history["train_loss"]) + 1)),
            y=history["train_loss"],
            mode="lines+markers",
            name="train_loss",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(history["val_loss"]) + 1)),
            y=history["val_loss"],
            mode="lines+markers",
            name="val_loss",
        )
    )

    fig.update_yaxes(title_text="Loss", type="log")
    fig.update_xaxes(title_text="Epoch")
    fig.update_layout(title=title, height=450, width=700)

    return fig


def animate_trajectories(
    trajectories,
    title="Diffusion Sampling",
    savepath="./sampling.gif",
    fps=30,
):
    duration_ms = max(20, int(round(1000 / fps)))

    frames = []

    for batch in trajectories:
        batch = _denormalize(batch)
        batch = batch.clamp(0.0, 1.0).detach().cpu()

        if batch.shape[1] == 1:
            batch = batch.repeat(1, 3, 1, 1)

        grid = make_grid(batch, nrow=4)
        grid = grid.permute(1, 2, 0)
        grid = (255 * grid).clamp(0, 255).to(torch.uint8).numpy()

        frames.append(grid)

    images = np.stack(frames)

    if savepath is not None:
        pil_images = [Image.fromarray(img) for img in images]
        pil_images[0].save(
            savepath,
            save_all=True,
            append_images=pil_images[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
            disposal=2,
        )

    fig = go.Figure(
        data=[go.Image(z=images[0])],
        frames=[
            go.Frame(data=[go.Image(z=images[i])], name=str(i))
            for i in range(len(images))
        ],
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
                args=[None, dict(
                    frame=dict(duration=duration_ms, redraw=True),
                    mode="immediate",
                    fromcurrent=True,
                    transition=dict(duration=0),
                )],
            )],
        )],
        sliders=[dict(steps=[
            dict(
                method="animate",
                label=str(i),
                args=[[str(i)], dict(
                    frame=dict(duration=0, redraw=True),
                    mode="immediate",
                    transition=dict(duration=0),
                )],
            )
            for i in range(len(images))
        ])],
    )

    return fig