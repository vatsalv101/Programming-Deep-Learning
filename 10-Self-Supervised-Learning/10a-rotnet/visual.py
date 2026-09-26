import math

import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

import data


def _image_grid(
    images,
    titles,
    title,
    cols=4,
    top_margin=80,
    vertical_spacing=0.04,
):
    n = images.shape[0]
    cols = min(cols, n)
    rows = math.ceil(n / cols)

    images = images.clamp(0.0, 1.0).detach().cpu()
    images = (255 * images.permute(0, 2, 3, 1)).to(torch.uint8).numpy()

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=titles,
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


def rotation_examples(images, rotation_labels):
    """
    Visualizes rotated images and their rotation labels.

    Args:
        images (Tensor): Image batch of shape (B, C, H, W).
        rotation_labels (Tensor): Rotation labels of shape (B,).
    
    Returns:
        fig (Figure): Plotly figure.
    """

    images = images.detach().cpu()
    rotation_labels = rotation_labels.detach().cpu()

    mean = torch.tensor(data.CIFAR10_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(data.CIFAR10_STD).view(1, 3, 1, 1)

    images = images * std + mean

    titles = [
        f"rotation: {90 * label.item()}°"
        for label in rotation_labels
    ]

    return _image_grid(
        images,
        titles,
        title="Rotation Prediction Examples",
        cols=4,
        top_margin=80,
        vertical_spacing=0.15,
    )


def show_training_stats(train_loss, test_acc, title):
    """
    Plots the training loss and test accuracy over epochs.

    Args:
        train_loss (list): List of training loss values for each epoch.
        test_acc (list): List of test accuracy values for each epoch.
        title (str): Title for the plot.

    Returns:
        fig (Figure): Plotly figure.
    """

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=["Training Loss", "Test Accuracy"],
    )

    fig.add_trace(
        go.Scatter(
            x=torch.arange(len(train_loss)),
            y=train_loss,
            mode="lines+markers",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=torch.arange(len(test_acc)),
            y=test_acc,
            mode="lines+markers",
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="loss", row=1, col=1)
    fig.update_yaxes(title_text="accuracy", row=2, col=1)
    fig.update_layout(showlegend=False, title=title)

    return fig
