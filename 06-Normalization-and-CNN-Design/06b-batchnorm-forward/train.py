import warnings

import torch
import torchvision
import torchvision.transforms.v2 as transforms

from torch import nn

try:
    import tqdm
except ModuleNotFoundError:
    class _SimpleTqdm:
        def __init__(self, iterable, desc=None):
            self.iterable = iterable
            self.desc = desc

        def __iter__(self):
            return iter(self.iterable)

        def set_description(self, desc):
            self.desc = desc

    class tqdm:
        @staticmethod
        def tqdm(iterable, desc=None):
            return _SimpleTqdm(iterable, desc=desc)


CIFAR10_MEAN = (0.5, 0.5, 0.5)
CIFAR10_STD = (0.5, 0.5, 0.5)


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        """
        Creates a baseline CNN without batch normalization layers.

        Args:
            num_classes (int): Number of output classes.
        """
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates logits of the baseline CNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32).

        Returns:
            logits (torch.Tensor): Raw class logits of shape
                (batch_size, num_classes).
        """

        features = self.features(x)
        features = torch.flatten(features, start_dim=1)
        logits = self.classifier(features)

        return logits

    def predict_class(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predicts class labels for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32).

        Returns:
            predicted_class (torch.Tensor): Predicted class indices.
        """

        logits = self.forward(x)
        predicted_class = torch.argmax(logits, dim=1)

        return predicted_class


class BatchNormCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        """
        Creates a CNN that uses batch normalization.

        Args:
            num_classes (int): Number of output classes.
        """
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates logits of the batch normalization CNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32).

        Returns:
            logits (torch.Tensor): Raw class logits of shape
                (batch_size, num_classes).
        """

        features = self.features(x)
        features = torch.flatten(features, start_dim=1)
        logits = self.classifier(features)

        return logits

    def predict_class(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predicts class labels for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32).

        Returns:
            predicted_class (torch.Tensor): Predicted class indices.
        """

        logits = self.forward(x)
        predicted_class = torch.argmax(logits, dim=1)

        return predicted_class


def _make_cifar10_loaders(
    batch_size: int,
    max_train_samples: int,
    max_test_samples: int,
    seed: int,
):
    transform_f = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*align should be passed as Python or NumPy boolean.*",
            category=Warning,
        )

        train_data = torchvision.datasets.CIFAR10(
            root="./data", train=True, transform=transform_f, download=True
        )
        test_data = torchvision.datasets.CIFAR10(
            root="./data", train=False, transform=transform_f, download=True
        )

    generator = torch.Generator().manual_seed(seed)

    train_indices = torch.randperm(len(train_data), generator=generator)[
        :max_train_samples
    ].tolist()
    test_indices = torch.randperm(len(test_data), generator=generator)[
        :max_test_samples
    ].tolist()

    train_subset = torch.utils.data.Subset(train_data, train_indices)
    test_subset = torch.utils.data.Subset(test_data, test_indices)

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    test_loader = torch.utils.data.DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


def _evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            predicted_class = model.predict_class(x)
            correct += (predicted_class == y).sum().item()
            total += y.numel()

    return correct / total


def _train_one_model(model, train_loader, test_loader, device, epochs, lr, weight_decay):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    train_loss_history = []
    test_acc_history = []

    pbar = tqdm.tqdm(
        range(epochs),
        desc="Train loss: -.---, Test accuracy: -.---",
    )

    for _ in pbar:
        model.train()
        epoch_loss = 0.0
        num_samples = 0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = y.shape[0]
            epoch_loss += loss.item() * batch_size
            num_samples += batch_size

        epoch_loss /= num_samples
        train_loss_history.append(epoch_loss)

        test_acc = _evaluate(model, test_loader, device)
        test_acc_history.append(test_acc)

        pbar.set_description(
            f"Train loss: {epoch_loss:.3f}, Test accuracy: {100 * test_acc:.2f} %"
        )

    return train_loss_history, test_acc_history


def train_short_comparison(
    epochs: int = 2,
    batch_size: int = 128,
    baseline_lr: float = 1e-3,
    baseline_weight_decay: float = 1e-6,
    batchnorm_lr: float = 3e-4,
    batchnorm_weight_decay: float = 0.0,
    max_train_samples: int = 4000,
    max_test_samples: int = 1000,
    seed: int = 0,
):
    """
    Trains a baseline CNN and a batch normalization CNN on a small CIFAR-10 subset.

    The run is intentionally short so it is practical on CPU. If CUDA is
    available, it is used automatically.

    Args:
        epochs (int): Number of training epochs for each model.
        batch_size (int): Number of samples in each batch.
        baseline_lr (float): Learning rate for Adam for the baseline model.
        baseline_weight_decay (float): Weight decay used by Adam for the baseline model.
        batchnorm_lr (float): Learning rate for Adam for the batch normalization model.
        batchnorm_weight_decay (float): Weight decay used by Adam for the batch normalization model.
        max_train_samples (int): Number of CIFAR-10 training samples to use.
        max_test_samples (int): Number of CIFAR-10 test samples to use.
        seed (int): Random seed for reproducible subset sampling.

    Returns:
        tuple[dict, dict]: Trained models and training histories.
    """
    torch.manual_seed(seed)

    train_loader, test_loader = _make_cifar10_loaders(
        batch_size=batch_size,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
        seed=seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    setups = {
        "baseline": (BaselineCNN(num_classes=10).to(device), baseline_lr, baseline_weight_decay),
        "batchnorm": (BatchNormCNN(num_classes=10).to(device), batchnorm_lr, batchnorm_weight_decay),
    }
    histories = {}

    for name, (model, lr, weight_decay) in setups.items():
        print(f"\nTraining {name}")
        train_loss, test_acc = _train_one_model(
            model,
            train_loader,
            test_loader,
            device=device,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
        )
        histories[name] = {
            "train_loss": train_loss,
            "test_acc": test_acc,
        }

    baseline_acc = histories["baseline"]["test_acc"][-1]
    batchnorm_acc = histories["batchnorm"]["test_acc"][-1]
    delta = batchnorm_acc - baseline_acc

    print(
        "\nFinal test accuracy: "
        f"baseline={100 * baseline_acc:.2f} %, "
        f"batchnorm={100 * batchnorm_acc:.2f} %, "
        f"delta={100 * delta:+.2f} percentage points"
    )

    models = {name: setup[0] for name, setup in setups.items()}
    return models, histories
