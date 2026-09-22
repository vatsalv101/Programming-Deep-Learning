import math
import warnings

import plotly.graph_objects as go
import torch
import torchvision
import torchvision.transforms.v2 as transforms
from plotly.subplots import make_subplots


def _load_test_data():
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ])

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".*align should be passed as Python or NumPy boolean.*", category=Warning)

        test_data = torchvision.datasets.CIFAR10(
            root="./data", train=False, transform=transform, download=True)

    return test_data


def _image_grid(images: torch.Tensor, titles: list[str], title: str, cols: int = 4, top_margin: int = 80, vertical_spacing: float = 0.04):
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


def cifar10_pictures(n: int = 12):
    """
    Plots random images from the CIFAR-10 test dataset.

    Args:
        n (int): Number of images to show.

    Returns:
        plotly.graph_objects.Figure: Figure containing image examples.
    """
    test_data = _load_test_data()
    class_names = test_data.classes

    n = min(n, len(test_data))
    indices = torch.randperm(len(test_data))[:n].tolist()

    images = []
    labels = []

    for index in indices:
        image, label = test_data[index]
        images.append(image)
        labels.append(label)

    images = torch.stack(images)
    titles = [f"label: {class_names[label]}" for label in labels]

    return _image_grid(images, titles, title="CIFAR-10 Examples", cols=4, top_margin=60, vertical_spacing=0.1)


def show_training_comparison(histories):
    """
    Plots training loss and test accuracy for multiple models.

    Args:
        histories (dict): Dictionary returned by cnn.train_short_comparison.

    Returns:
        plotly.graph_objects.Figure: Figure with training curves.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Training Loss", "Test Accuracy"],
    )

    for name, history in histories.items():
        epochs = list(range(1, len(history["train_loss"]) + 1))

        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history["train_loss"],
                mode="lines+markers",
                name=f"{name} loss",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history["test_acc"],
                mode="lines+markers",
                name=f"{name} accuracy",
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(title_text="epoch", row=1, col=1)
    fig.update_xaxes(title_text="epoch", row=1, col=2)
    fig.update_yaxes(title_text="loss", row=1, col=1)
    fig.update_yaxes(title_text="accuracy", row=1, col=2)
    fig.update_layout(title="Short CNN Comparison")

    return fig
