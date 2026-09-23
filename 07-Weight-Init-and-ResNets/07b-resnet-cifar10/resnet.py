import torch
from torch import nn

from blocks import BasicBlock, make_block_group
from data import CIFAR10_MEAN, CIFAR10_STD, make_cifar10_loaders
from shortcuts import ProjectionShortcut, ZeroPadShortcut


class ResNet(nn.Module):

    """
    CIFAR-style ResNet.
    """

    def __init__(
        self,
        num_blocks=(3, 3, 3),
        channels=(16, 32, 64),
        num_classes=10,
        shortcut="zero_pad",
    ):
        """
        Args:
            num_blocks (tuple[int, int, int]): Number of blocks in the three
                block groups.
            channels (tuple[int, int, int]): Channel counts for the block
                groups.
            num_classes (int): Number of output classes.
            shortcut (str): Shortcut type, "zero_pad" or "projection".
        """

        super().__init__()

        ################################################################
        # TODO

        n1, n2, n3 = num_blocks
        c1, c2, c3 = channels

        self.proj = nn.Sequential(
            nn.Conv2d(3, c1, 1, stride=2, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
        )

        self.b1 = make_block_group(c1, c1, n1, 1, shortcut)
        self.b2 = make_block_group(c1, c2, n2, 2, shortcut)
        self.b3 = make_block_group(c2, c3, n3, 2, shortcut)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        self.out = nn.Linear(c3, num_classes)
        
        ################################################################

    def forward(self, x):
        """
        Calculates raw class logits.
        """

        ################################################################
        # TODO

        logits = self.proj(x)
        logits = self.b1(logits)
        logits = self.b2(logits)
        logits = self.b3(logits)
        logits = self.avg(logits)
        logits = torch.flatten(logits, 1)
        logits = self.out(logits)
        ################################################################

        return logits

    def predict_class(self, x):
        """
        Predicts class indices.
        """

        ################################################################
        # TODO

        logits = self(x)
        predicted_class = torch.argmax(logits, dim=-1)

        ################################################################

        return predicted_class





# Modify later if you want to add more ResNet configurations.
def resnet20(num_classes=10, shortcut="zero_pad"):
    """
    Creates the CIFAR ResNet-20 configuration from the paper.
    """

    return ResNet(
        num_blocks=(3, 3, 3),
        channels=(16, 32, 64),
        num_classes=num_classes,
        shortcut=shortcut,
    )
















































from training import (  # noqa: E402
    evaluate_accuracy,
    initialize,
    set_learning_rate,
    step_learning_rate,
    train,
    train_one_epoch,
    training_step,
)


__all__ = [
    "BasicBlock",
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    "ProjectionShortcut",
    "ResNet",
    "ZeroPadShortcut",
    "evaluate_accuracy",
    "initialize",
    "make_block_group",
    "make_cifar10_loaders",
    "resnet20",
    "set_learning_rate",
    "step_learning_rate",
    "train",
    "train_one_epoch",
    "training_step",
]
