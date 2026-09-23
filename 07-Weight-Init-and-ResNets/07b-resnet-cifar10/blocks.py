from torch import nn

from shortcuts import ProjectionShortcut, ZeroPadShortcut


class BasicBlock(nn.Module):
    """
    Basic residual block for CIFAR-style ResNets.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        shortcut="zero_pad",
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the first convolution in the block.
            shortcut (str): Shortcut type, "zero_pad" or "projection".
        """

        super().__init__()

        ################################################################
        # TODO

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU()

        if in_channels == out_channels and stride == 1:
            self.skip = nn.Identity()
        elif shortcut == "zero_pad":
            self.skip = ZeroPadShortcut(out_channels, stride)
        elif shortcut == "projection":
            self.skip = ProjectionShortcut(in_channels, out_channels, stride)
        else:
            raise ValueError("error")

        ################################################################

    def forward(self, x):
        """
        Applies the residual block.
        """

        ################################################################
        # TODO

        out = self.conv1(x)
        out = self.relu(out + self.skip(x))        

        ################################################################

        return out


def make_block_group(in_channels, out_channels, num_blocks, stride, shortcut):
    """
    Creates a group of residual blocks with the same output shape.
    """

    ################################################################
    # TODO

    blocks = [BasicBlock(in_channels, out_channels, stride, shortcut)]

    for _ in range(num_blocks-1):
        blocks.append(BasicBlock(out_channels, out_channels, 1, shortcut))

    block_group = nn.Sequential(*blocks)
    ################################################################

    return block_group
