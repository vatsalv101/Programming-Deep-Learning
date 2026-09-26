import math

import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

import data


def _unnormalize(images):
    mean = torch.tensor(data.CIFAR10_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(data.CIFAR10_STD).view(1, 3, 1, 1)
    return images.detach().cpu() * std + mean


def _image_grid(images, titles, title, cols=4, top_margin=80, vertical_spacing=0.04):
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


def two_view_examples(view1, view2, max_images=4):
    """
    Visualizes BYOL views as a two-row grid.

    Top row: first view of each image.
    Bottom row: second view of the same images.
    """

    n = min(max_images, view1.shape[0], view2.shape[0])

    view1 = _unnormalize(view1[:n])
    view2 = _unnormalize(view2[:n])

    images = torch.cat([view1, view2], dim=0)
    titles = [f"image {index}: view 1" for index in range(n)]
    titles += [f"image {index}: view 2" for index in range(n)]

    return _image_grid(
        images,
        titles,
        title="BYOL Augmentation Pairs",
        cols=n,
        top_margin=80,
        vertical_spacing=0.16,
    )


def show_loss_curve(loss_history, title):
    """
    Plots a training loss curve.
    """

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=torch.arange(len(loss_history)),
            y=loss_history,
            mode="lines+markers",
        )
    )
    fig.update_xaxes(title_text="epoch")
    fig.update_yaxes(title_text="loss")
    fig.update_layout(showlegend=False, title=title)
    return fig


def show_training_stats(train_loss, test_acc, title):
    """
    Plots linear-probe training loss and test accuracy over epochs.
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
