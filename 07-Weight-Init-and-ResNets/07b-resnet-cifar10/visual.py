import math
import warnings

import plotly.graph_objects as go
import torch
import torchvision
import torchvision.transforms.v2 as transforms
from plotly.subplots import make_subplots


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _load_test_data():
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*align should be passed as Python or NumPy boolean.*",
            category=Warning,
        )

        test_data = torchvision.datasets.CIFAR10(
            root="./data",
            train=False,
            transform=transform,
            download=True,
        )

    return test_data


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


def cifar10_pictures(n=12):
    """
    Plots random images from the CIFAR-10 test dataset.
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

    return _image_grid(
        images,
        titles,
        title="CIFAR-10 Examples",
        cols=4,
        top_margin=60,
        vertical_spacing=0.1,
    )


def cifar10_predictions(model, n=8):
    """
    Plots model predictions on random CIFAR-10 test images.
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

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)

    with torch.no_grad():
        normalized = (images - mean) / std
        predicted = model.predict_class(normalized.to(device)).detach().cpu().tolist()

    if was_training:
        model.train()

    titles = [
        f"prediction: {class_names[pred]}<br>label: {class_names[label]}"
        for pred, label in zip(predicted, labels)
    ]

    return _image_grid(
        images,
        titles,
        title="CIFAR-10 Predictions",
        cols=4,
        top_margin=80,
        vertical_spacing=0.2,
    )


def show_training_stats(train_loss, test_acc):
    """
    Plots the training loss and test accuracy over epochs.
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
    fig.update_layout(showlegend=False, title="ResNet Training")

    return fig
