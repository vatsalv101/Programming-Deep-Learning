import warnings
from pathlib import Path

import torch
import torchvision
import torchvision.transforms.v2 as transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# use the data root for downloading the dataset to avoid multiple downloads
data_path = Path.cwd().resolve().parent.parent / "data"
data_path.mkdir(parents=True, exist_ok=True)
data_root = str(data_path)


def make_cifar10_loaders(batch_size, num_workers=0):
    """
    Creates CIFAR-10 train and test data loaders.

    Args:
        batch_size (int): Batch size for the data loaders.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        train_loader (DataLoader): DataLoader for the CIFAR-10 training set.
        test_loader (DataLoader): DataLoader for the CIFAR-10 test set.
    """

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*align should be passed as Python or NumPy boolean.*",
            category=Warning,
        )

        train_data = torchvision.datasets.CIFAR10(
            root=data_root, train=True, transform=train_transform, download=True)
        test_data = torchvision.datasets.CIFAR10(
            root=data_root, train=False, transform=test_transform, download=True)

    kwargs = {"num_workers": num_workers}
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader
