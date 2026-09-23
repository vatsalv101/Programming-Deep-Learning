import warnings

import torch
import torchvision
import torchvision.transforms.v2 as transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def make_cifar10_loaders(batch_size, num_workers=0):
    """
    Creates CIFAR-10 train and test data loaders.

    Data loading and transformations are provided for this exercise.
    """

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )

    test_transform = transforms.Compose(
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
            root="./data",
            train=True,
            transform=train_transform,
            download=True,
        )
        test_data = torchvision.datasets.CIFAR10(
            root="./data",
            train=False,
            transform=test_transform,
            download=True,
        )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader
