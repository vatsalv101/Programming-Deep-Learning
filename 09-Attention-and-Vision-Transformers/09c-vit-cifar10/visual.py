import math
import warnings
from pathlib import Path

import plotly.graph_objects as go
import torch
import torchvision
import torchvision.transforms.v2 as transforms
from plotly.subplots import make_subplots


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

data_root = Path.cwd().resolve().parent.parent / "data"
data_root.mkdir(parents=True, exist_ok=True)

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
            root=str(data_root),
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


def _find_last_attention_module(model):

    """
    Finds the last MultiHeadSelfAttention module, preferring the mha inside the last EncoderBlock.
    """

    encoder_blocks = [
        module
        for module in model.modules()
        if module.__class__.__name__ == "EncoderBlock"
    ]

    if len(encoder_blocks) > 0:
        last_block = encoder_blocks[-1]
        attention_module = getattr(last_block, "mha", None)
        if attention_module is not None:
            return attention_module

    attention_modules = [
        module
        for module in model.modules()
        if module.__class__.__name__ == "MultiHeadSelfAttention"
    ]

    if len(attention_modules) == 0:
        return None

    return attention_modules[-1]


def cifar10_attention(model, n=5):
    
    """
    Visualizes learned attention weights on random CIFAR-10 images.
    """

    test_data = _load_test_data()

    attention_module = _find_last_attention_module(model)
    if attention_module is None:
        raise ValueError(
            "Could not find a MultiHeadSelfAttention module inside the model. "
            "Pass a trained ViT-style model that uses EncoderBlock modules."
        )

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)

    correct_images = []
    candidate_indices = torch.randperm(len(test_data)).tolist()

    try:
        with torch.no_grad():
            for start in range(0, len(candidate_indices), 64):
                batch_indices = candidate_indices[start : start + 64]
                batch_images = []
                batch_labels = []

                for index in batch_indices:
                    image, label = test_data[index]
                    batch_images.append(image)
                    batch_labels.append(label)

                batch_images = torch.stack(batch_images)
                normalized = (batch_images - mean) / std
                predictions = model.predict_class(normalized.to(device)).detach().cpu()

                for image, label, prediction in zip(batch_images, batch_labels, predictions):
                    if int(prediction) == label:
                        correct_images.append(image)
                        if len(correct_images) == n:
                            break

                if len(correct_images) == n:
                    break
    finally:
        if was_training:
            model.train()

    if len(correct_images) < n:
        raise RuntimeError(
            f"Could only find {len(correct_images)} correctly classified CIFAR-10 images. "
            f"Need {n} for the visualization."
        )

    images = torch.stack(correct_images)

    model.eval()

    captured_output = {}

    def _capture_attention_output(module, inputs, output):
        captured_output["attn_weights"] = output[1].detach()

    handle = attention_module.register_forward_hook(_capture_attention_output)

    try:
        with torch.no_grad():
            normalized = (images - mean) / std
            _ = model(normalized.to(device))
    finally:
        handle.remove()
        if was_training:
            model.train()

    if "attn_weights" not in captured_output:
        raise RuntimeError(
            "The attention hook did not capture attention weights. Make sure the model actually calls the attention module."
        )

    attention_weights = captured_output["attn_weights"]
    seq_len = attention_weights.shape[-1]

    if seq_len <= 1:
        raise ValueError("Attention map needs at least one CLS token and one patch token.")

    token_weights = attention_weights.mean(dim=1)[:, 0, 1:]

    if token_weights.shape[-1] == 0:
        raise ValueError(
            "Could not extract a CLS attention map over patch tokens. "
            "This helper expects a ViT-style model with a CLS token followed by patch tokens."
        )

    grid_size = int(math.isqrt(token_weights.shape[-1]))
    if grid_size * grid_size != token_weights.shape[-1]:
        raise ValueError(
            "Attention weights do not form a square patch grid. "
            "This helper expects a ViT-style model with square patch layout."
        )

    patch_height = images.shape[-2] // grid_size
    patch_width = images.shape[-1] // grid_size

    highlighted_images = []
    for image, weights in zip(images, token_weights):
        weight_grid = weights.reshape(grid_size, grid_size)
        weight_grid = weight_grid - weight_grid.min()
        if float(weight_grid.max()) > 0:
            weight_grid = weight_grid / weight_grid.max()

        highlighted = image.clone()
        for row in range(grid_size):
            for col in range(grid_size):
                scale = 0.35 + 0.65 * float(weight_grid[row, col])
                row_start = row * patch_height
                row_end = (row + 1) * patch_height
                col_start = col * patch_width
                col_end = (col + 1) * patch_width
                highlighted[:, row_start:row_end, col_start:col_end] *= scale

        highlighted_images.append(highlighted)

    highlighted_images = torch.stack(highlighted_images)

    fig = make_subplots(
        rows=n,
        cols=2,
        horizontal_spacing=0.03,
        vertical_spacing=0.08,
    )

    original_images = images.clamp(0.0, 1.0).detach().cpu()
    highlighted_images = highlighted_images.clamp(0.0, 1.0).detach().cpu()

    original_images = (255 * original_images.permute(0, 2, 3, 1)).to(torch.uint8).numpy()
    highlighted_images = (255 * highlighted_images.permute(0, 2, 3, 1)).to(torch.uint8).numpy()

    for index in range(n):
        fig.add_trace(go.Image(z=original_images[index]), row=index + 1, col=1)
        fig.add_trace(go.Image(z=highlighted_images[index]), row=index + 1, col=2)

    fig.add_annotation(
        text="original",
        x=0.25,
        y=1.08,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16),
    )
    fig.add_annotation(
        text="attention",
        x=0.75,
        y=1.08,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16),
    )

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        title=dict(text="CIFAR-10 Attention Visualization", y=0.98),
        showlegend=False,
        height=max(190 * n, 240),
        width=560,
        margin=dict(l=20, r=20, t=110, b=20),
    )

    return fig


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
    fig.update_layout(showlegend=False, title="ViT Training")

    return fig
