import warnings
from pathlib import Path

import torch
import torchvision
import torchvision.transforms.v2 as transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# use the data root for downloading the dataset to avoid multiple downloads
# when several exercises are stored in neighboring folders
_data_path = Path.cwd().resolve().parent.parent / "data"
_data_path.mkdir(parents=True, exist_ok=True)
data_root = str(_data_path)


class TwoViewTransform:
    """
    Creates two augmented views of one image.

    The augmentation probabilities follow the BYOL paper, adapted from
    ImageNet resolution to CIFAR-10 resolution.
    """

    def __init__(self):
        ################################################################
        # TODO
        
        self.transform_1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(32, scale = (0.08, 1.0), ratio = (3/4, 4/3)),
                transforms.RandomHorizontalFlip(p= 0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.2,0.1)], p = 0.8),
                transforms.RandomGrayscale(p=.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size = 3, sigma = (0.1, 2))], p = 1.0),
                transforms.RandomSolarize(threshold = 0.5, p = 0),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale = True),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )

        self.transform_2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(32, scale = (0.08, 1.0), ratio = (3/4, 4/3)),
                transforms.RandomHorizontalFlip(p= 0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.2,0.1)], p = 0.8),
                transforms.RandomGrayscale(p=.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size = 3, sigma = (0.1, 2))], p = 0.1),
                transforms.RandomSolarize(threshold = 0.5, p = 0.2),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale = True),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )


        ################################################################

    def __call__(self, image):
        ################################################################
        # TODO
        view_1 = self.transform_1(image)
        view_2 = self.transform_2(image)


        
        ################################################################
        return view_1, view_2


def make_linear_eval_transforms():
    """
    Creates train and test transforms for supervised linear evaluation.
    """

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
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

    return train_transform, test_transform


def _loader_kwargs(num_workers):
    kwargs = {"num_workers": num_workers}
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


def make_byol_loader(batch_size, num_workers=0):
    """
    Creates the CIFAR-10 training loader for BYOL pretraining.

    The original CIFAR-10 labels are still returned by the dataset but are not
    used by BYOL.
    """

    transform = TwoViewTransform()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*align should be passed as Python or NumPy boolean.*",
            category=Warning,
        )

        train_data = torchvision.datasets.CIFAR10(
            root=data_root,
            train=True,
            transform=transform,
            download=True,
        )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **_loader_kwargs(num_workers),
    )

    return train_loader


def make_cifar10_loaders(batch_size, num_workers=0):
    """
    Creates CIFAR-10 train and test loaders for supervised linear evaluation.
    """

    train_transform, test_transform = make_linear_eval_transforms()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*align should be passed as Python or NumPy boolean.*",
            category=Warning,
        )

        train_data = torchvision.datasets.CIFAR10(
            root=data_root,
            train=True,
            transform=train_transform,
            download=True,
        )
        test_data = torchvision.datasets.CIFAR10(
            root=data_root,
            train=False,
            transform=test_transform,
            download=True,
        )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        **_loader_kwargs(num_workers),
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        **_loader_kwargs(num_workers),
    )

    return train_loader, test_loader
