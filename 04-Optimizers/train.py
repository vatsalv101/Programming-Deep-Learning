import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt

from model import MLP


def make_dataset(seed=42):

    """
    Creates a small synthetic classification dataset.

    The dataset is designed to stress optimizers rather than model capacity.

    The class signal is present in small dense and sparse features, while
    some larger nuisance features make a single global learning rate harder
    to tune. Adaptive optimizers should handle the scale mismatch better.
    """

    points_per_class = 120
    num_classes = 3
    input_dim = 32

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    num_examples = points_per_class * num_classes

    X = torch.zeros(num_examples, input_dim)
    y = torch.zeros(num_examples, dtype=torch.long)

    # ----------------------------------------------------------------
    # 1. Small-scale dense informative features
    # ----------------------------------------------------------------

    dense_dim = 6

    dense_prototypes = torch.tensor([
        [ 1.0,  0.0,  1.0, -1.0,  0.5, -0.5],
        [-0.5,  0.9, -1.0,  0.5,  1.0, -0.5],
        [-0.5, -0.9,  0.5,  0.5, -1.0,  1.0],
    ])

    dense_signal_scale = 0.20
    dense_noise_scale = 0.05

    for class_id in range(num_classes):

        ix = range(
            points_per_class * class_id,
            points_per_class * (class_id + 1)
        )

        y[ix] = class_id

        X[ix, :dense_dim] = (
            dense_signal_scale * dense_prototypes[class_id]
            + dense_noise_scale * torch.randn(
                points_per_class,
                dense_dim,
                generator=generator
            )
        )

    # ----------------------------------------------------------------
    # 2. Small-scale sparse informative features
    # ----------------------------------------------------------------

    num_sparse_per_class = 4
    num_sparse = num_classes * num_sparse_per_class

    sparse_start = dense_dim
    sparse_end = sparse_start + num_sparse

    inactive_probability = 0.03
    active_probability = 0.60
    sparse_signal_scale = 0.20

    sparse = torch.zeros(num_examples, num_sparse)

    for class_id in range(num_classes):

        class_mask = y == class_id

        for j in range(num_sparse_per_class):

            feature_id = class_id * num_sparse_per_class + j

            probabilities = torch.full(
                (num_examples,),
                inactive_probability
            )

            probabilities[class_mask] = active_probability

            sparse[:, feature_id] = torch.bernoulli(
                probabilities,
                generator=generator
            )

    sparse = sparse - sparse.mean(dim=0, keepdim=True)
    X[:, sparse_start:sparse_end] = sparse_signal_scale * sparse

    # ----------------------------------------------------------------
    # 3. Moderate-scale nuisance features
    # ----------------------------------------------------------------

    nuisance_start = sparse_end
    nuisance_dim = input_dim - nuisance_start

    nuisance_scale = 3.0

    nuisance = torch.randn(
        num_examples,
        nuisance_dim,
        generator=generator
    )

    nuisance = nuisance - nuisance.mean(dim=0, keepdim=True)

    X[:, nuisance_start:] = nuisance_scale * nuisance

    # Shuffle examples.
    perm = torch.randperm(num_examples, generator=generator)

    X = X[perm]
    y = y[perm]

    return X, y


def make_dataloaders(batch_size=64, test_fraction=0.50, seed=42):

    """
    Creates train and test dataloaders.

    Args:
        batch_size    (int): Batch size.
        test_fraction (float): Fraction used for testing.
        seed          (int): Random seed.

    Returns:
        train_loader, test_loader
    """

    X, y = make_dataset(seed=seed)

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(X.shape[0], generator=generator)

    X = X[perm]
    y = y[perm]

    num_test = int(test_fraction * X.shape[0])

    X_test = X[:num_test]
    y_test = y[:num_test]

    X_train = X[num_test:]
    y_train = y[num_test:]

    train_dataset = data.TensorDataset(X_train, y_train)
    test_dataset = data.TensorDataset(X_test, y_test)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader


def compute_accuracy(model, data_loader, device):

    """
    Computes classification accuracy.

    Args:
        model       (nn.Module): Model.
        data_loader (DataLoader): Data loader.
        device      (torch.device): Device.

    Returns:
        float: Accuracy in [0, 1].
    """

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in data_loader:

            X = X.to(device)
            y = y.to(device)

            scores = model(X)
            predictions = torch.argmax(scores, dim=1)

            correct += torch.sum(predictions == y).item()
            total += y.numel()

    return correct / total


def compute_parameter_norm(model):

    """
    Computes the L2 norm of all trainable parameters.

    Args:
        model (nn.Module): Model.

    Returns:
        float: Parameter norm.
    """

    total = 0.0

    with torch.no_grad():
        for param in model.parameters():
            total += torch.sum(param ** 2).item()

    return total ** 0.5


def run_experiment(
    optimizer_name,
    zero_grad_fn,
    optimizer_step_fn,
    init_state_fn=None,
    optimizer_kwargs=None,
    epochs=60,
    batch_size=64,
    seed=42,
    device=None,
    print_every=5
):

    """
    Runs one training experiment with the selected optimizer.

    Args:
        optimizer_name    (str): Name used for printing.
        zero_grad_fn      (callable): Function that zeros gradients.
        optimizer_step_fn (callable): Optimizer update function.
        init_state_fn     (callable or None): Function that initializes optimizer state.
        optimizer_kwargs  (dict or None): Keyword arguments passed to the optimizer step.
        epochs            (int): Number of training epochs.
        batch_size        (int): Batch size.
        seed              (int): Random seed.
        device            (torch.device or None): Device.
        print_every       (int): Print progress every print_every epochs.

    Returns:
        dict: Training loss, test accuracy, and parameter norm histories.
    """

    if device is None:
        device = torch.device("cpu")

    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    torch.manual_seed(seed)

    train_loader, test_loader = make_dataloaders(
        batch_size=batch_size,
        seed=seed
    )

    model = MLP().to(device)
    loss_fn = nn.CrossEntropyLoss()

    params = list(model.parameters())

    state = None
    if init_state_fn is not None:
        state = init_state_fn(params)

    results = {
        "train_loss": [],
        "test_acc": [],
        "param_norm": []
    }

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0
        total_examples = 0

        for X, y in train_loader:

            X = X.to(device)
            y = y.to(device)

            scores = model(X)
            loss = loss_fn(scores, y)

            zero_grad_fn(params)
            loss.backward()

            state = optimizer_step_fn(
                params,
                state,
                **optimizer_kwargs
            )

            total_loss += loss.item() * y.shape[0]
            total_examples += y.shape[0]

        train_loss = total_loss / total_examples
        test_acc = compute_accuracy(model, test_loader, device)
        param_norm = compute_parameter_norm(model)

        results["train_loss"].append(train_loss)
        results["test_acc"].append(test_acc)
        results["param_norm"].append(param_norm)

        should_print = (
            epoch == 0
            or (epoch + 1) % print_every == 0
            or epoch == epochs - 1
        )

        if should_print:
            print(
                f"{optimizer_name} | "
                f"epoch {epoch + 1:02d}/{epochs} | "
                f"loss {train_loss:.4f} | "
                f"acc {100 * test_acc:.2f}% | "
                f"norm {param_norm:.2f}"
            )

    return results


def plot_results(results, names):

    """
    Plots training loss, test accuracy, and parameter norm.

    Args:
        results (List[dict]): Result dictionaries.
        names   (List[str]): Optimizer names.
    """

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    plot_specs = [
        ("train_loss", "Training Loss", "Loss"),
        ("test_acc", "Test Accuracy", "Accuracy"),
        ("param_norm", "Parameter Norm", "L2 Norm")
    ]

    for ax, (key, title, ylabel) in zip(axes, plot_specs):

        for result, name in zip(results, names):

            if name.lower() == "adamw":
                linestyle = "--"
                linewidth = 2.3
            else:
                linestyle = "-"
                linewidth = 1.8

            ax.plot(
                range(1, len(result[key]) + 1),
                result[key],
                label=name,
                linestyle=linestyle,
                linewidth=linewidth
            )

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)

        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
