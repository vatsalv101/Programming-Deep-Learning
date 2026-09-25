import torch
import torchvision
import torchvision.transforms.v2 as transforms

import data
import patchify


class _DummyCIFAR10(torch.utils.data.Dataset):

    """
    Lightweight stand-in for torchvision.datasets.CIFAR10 used by tests.
    """

    init_calls = []

    def __init__(self, root, train, transform, download):
        self.root = root
        self.train = train
        self.transform = transform
        self.download = download
        self.classes = [f"class_{index}" for index in range(10)]

        self.images = torch.randint(
            low=0,
            high=256,
            size=(24, 3, 32, 32),
            dtype=torch.uint8,
        )
        self.labels = torch.arange(24, dtype=torch.int64) % 10

        _DummyCIFAR10.init_calls.append(
            {
                "root": root,
                "train": train,
                "transform": transform,
                "download": download,
            }
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = int(self.labels[index])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def _with_dummy_cifar10_loader(batch_size=8, num_workers=0):
    original_cifar10 = torchvision.datasets.CIFAR10
    _DummyCIFAR10.init_calls = []

    try:
        torchvision.datasets.CIFAR10 = _DummyCIFAR10
        train_loader, test_loader = data.make_cifar10_loaders(
            batch_size=batch_size,
            num_workers=num_workers,
        )
    finally:
        torchvision.datasets.CIFAR10 = original_cifar10

    return train_loader, test_loader


def test_make_cifar10_loaders_transforms():

    """
    Verifies that train/test transformations are correctly configured.
    """

    try:
        train_loader, test_loader = _with_dummy_cifar10_loader(batch_size=8)
    except Exception as error:
        print(f"FAIL: make_cifar10_loaders raised an error ({error}).")
        return False

    if len(_DummyCIFAR10.init_calls) != 2:
        print("FAIL: Expected exactly two CIFAR10 dataset initializations (train and test).")
        return False

    train_call, test_call = _DummyCIFAR10.init_calls

    if train_call["train"] is not True or test_call["train"] is not False:
        print("FAIL: CIFAR10 should be initialized with train=True for train set and train=False for test set.")
        return False

    if not train_call["download"] or not test_call["download"]:
        print("FAIL: CIFAR10 datasets should be initialized with download=True.")
        return False

    train_transform = train_loader.dataset.transform
    test_transform = test_loader.dataset.transform

    if not isinstance(train_transform, transforms.Compose):
        print("FAIL: train_transform should be torchvision.transforms.v2.Compose.")
        return False

    if not isinstance(test_transform, transforms.Compose):
        print("FAIL: test_transform should be torchvision.transforms.v2.Compose.")
        return False

    train_ops = train_transform.transforms
    test_ops = test_transform.transforms

    expected_train_types = [
        transforms.RandomCrop,
        transforms.RandomHorizontalFlip,
        transforms.ToImage,
        transforms.ToDtype,
        transforms.Normalize,
    ]
    expected_test_types = [
        transforms.ToImage,
        transforms.ToDtype,
        transforms.Normalize,
    ]

    if len(train_ops) != len(expected_train_types):
        print(
            "FAIL: train_transform has the wrong number of operations. "
            f"Got {len(train_ops)}, expected {len(expected_train_types)}."
        )
        return False

    if len(test_ops) != len(expected_test_types):
        print(
            "FAIL: test_transform has the wrong number of operations. "
            f"Got {len(test_ops)}, expected {len(expected_test_types)}."
        )
        return False

    if not all(isinstance(op, expected) for op, expected in zip(train_ops, expected_train_types)):
        print("FAIL: train_transform operations are not in the expected order.")
        return False

    if not all(isinstance(op, expected) for op, expected in zip(test_ops, expected_test_types)):
        print("FAIL: test_transform operations are not in the expected order.")
        return False

    normalize_train = train_ops[-1]
    normalize_test = test_ops[-1]

    if tuple(float(value) for value in normalize_train.mean) != tuple(data.CIFAR10_MEAN):
        print("FAIL: train_transform.Normalize does not use CIFAR10_MEAN.")
        return False

    if tuple(float(value) for value in normalize_train.std) != tuple(data.CIFAR10_STD):
        print("FAIL: train_transform.Normalize does not use CIFAR10_STD.")
        return False

    if tuple(float(value) for value in normalize_test.mean) != tuple(data.CIFAR10_MEAN):
        print("FAIL: test_transform.Normalize does not use CIFAR10_MEAN.")
        return False

    if tuple(float(value) for value in normalize_test.std) != tuple(data.CIFAR10_STD):
        print("FAIL: test_transform.Normalize does not use CIFAR10_STD.")
        return False

    print("PASS: make_cifar10_loaders transformation pipelines are correct.")
    return True


def test_make_cifar10_loaders_outputs():

    """
    Verifies output shapes/dtypes and loader shuffle behavior.
    """

    try:
        batch_size = 6
        train_loader, test_loader = _with_dummy_cifar10_loader(batch_size=batch_size)
    except Exception as error:
        print(f"FAIL: make_cifar10_loaders raised an error ({error}).")
        return False

    train_images, train_labels = next(iter(train_loader))
    test_images, test_labels = next(iter(test_loader))

    if train_images.shape != (batch_size, 3, 32, 32):
        print(
            "FAIL: train_loader returned wrong image shape. "
            f"Got {train_images.shape}, expected {(batch_size, 3, 32, 32)}."
        )
        return False

    if test_images.shape != (batch_size, 3, 32, 32):
        print(
            "FAIL: test_loader returned wrong image shape. "
            f"Got {test_images.shape}, expected {(batch_size, 3, 32, 32)}."
        )
        return False

    if train_images.dtype != torch.float32:
        print(f"FAIL: train images dtype should be torch.float32, got {train_images.dtype}.")
        return False

    if test_images.dtype != torch.float32:
        print(f"FAIL: test images dtype should be torch.float32, got {test_images.dtype}.")
        return False

    if train_labels.dtype != torch.int64:
        print(f"FAIL: train labels dtype should be torch.int64, got {train_labels.dtype}.")
        return False

    if test_labels.dtype != torch.int64:
        print(f"FAIL: test labels dtype should be torch.int64, got {test_labels.dtype}.")
        return False

    if not torch.isfinite(train_images).all() or not torch.isfinite(test_images).all():
        print("FAIL: loader outputs contain non-finite values.")
        return False

    if not isinstance(train_loader.sampler, torch.utils.data.RandomSampler):
        print("FAIL: train_loader should use random sampling (shuffle=True).")
        return False

    if not isinstance(test_loader.sampler, torch.utils.data.SequentialSampler):
        print("FAIL: test_loader should use sequential sampling (shuffle=False).")
        return False

    print("PASS: make_cifar10_loaders outputs and loader settings are correct.")
    return True

def test_data_loaders():

    """
    Runs all data loader tests.
    """
    test_make_cifar10_loaders_transforms()
    test_make_cifar10_loaders_outputs()


def test_create_patch_sequence_shape_and_content():

    """
    Verifies shape and patch content on a simple toy example,
    without enforcing a specific sequence ordering.
    """

    x = torch.tensor(
        [
            [
                [
                    [0.0, 1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0, 7.0],
                    [8.0, 9.0, 10.0, 11.0],
                    [12.0, 13.0, 14.0, 15.0],
                ]
            ]
        ]
    )

    try:
        sequence = patchify.create_patch_sequence(x, patch_size=2)
    except Exception as error:
        print(f"FAIL: create_patch_sequence raised an error ({error}).")
        return False

    if sequence.shape != (1, 4, 4):
        print(f"FAIL: create_patch_sequence returned shape {sequence.shape}, expected (1, 4, 4).")
        return False

    expected = torch.tensor(
        [
            [
                [0.0, 1.0, 4.0, 5.0],
                [2.0, 3.0, 6.0, 7.0],
                [8.0, 9.0, 12.0, 13.0],
                [10.0, 11.0, 14.0, 15.0],
            ]
        ]
    )

    actual_patch_rows = sorted(tuple(row.tolist()) for row in sequence[0])
    expected_patch_rows = sorted(tuple(row.tolist()) for row in expected[0])

    if actual_patch_rows != expected_patch_rows:
        print("FAIL: create_patch_sequence extracted incorrect patch values.")
        return False

    print("PASS: create_patch_sequence shape and patch content are correct.")
    return True


def test_create_patch_sequence_batch_and_dtype():

    """
    Verifies output shape and dtype preservation for batched RGB inputs.
    """

    x = torch.randn(3, 3, 32, 32, dtype=torch.float32)

    try:
        sequence = patchify.create_patch_sequence(x, patch_size=4)
    except Exception as error:
        print(f"FAIL: create_patch_sequence raised an error ({error}).")
        return False

    expected_shape = (3, (32 // 4) * (32 // 4), 3 * 4 * 4)
    if sequence.shape != expected_shape:
        print(f"FAIL: create_patch_sequence returned shape {sequence.shape}, expected {expected_shape}.")
        return False

    if sequence.dtype != x.dtype:
        print(f"FAIL: create_patch_sequence should preserve dtype. Got {sequence.dtype}, expected {x.dtype}.")
        return False

    if not torch.isfinite(sequence).all():
        print("FAIL: create_patch_sequence output contains non-finite values.")
        return False

    print("PASS: create_patch_sequence handles batched RGB inputs correctly.")
    return True


def test_create_patch_sequence_invalid_patch_size():

    """
    Verifies that invalid patch sizes are rejected.
    """
    x = torch.randn(2, 3, 32, 32)

    try:
        patchify.create_patch_sequence(x, patch_size=5)
    except AssertionError:
        print("PASS: create_patch_sequence rejects non-divisible patch sizes.")
        return True
    except Exception as error:
        print(f"FAIL: create_patch_sequence raised the wrong exception type ({error}).")
        return False

    print("FAIL: create_patch_sequence should raise for non-divisible patch sizes.")
    return False


def test_patchify():

    """
    Runs all patchify tests.
    """
    
    test_create_patch_sequence_batch_and_dtype()
    test_create_patch_sequence_shape_and_content()
    test_create_patch_sequence_invalid_patch_size()
