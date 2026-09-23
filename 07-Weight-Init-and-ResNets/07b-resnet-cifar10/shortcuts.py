import torch
from torch import nn


class ZeroPadShortcut(nn.Module):

    """
    Parameter-free CIFAR ResNet shortcut from option A.
    """

    def __init__(self, out_channels, stride):
        
        """
        Args:
            out_channels (int): Number of output channels after padding.
            stride (int): Spatial stride used by the residual branch.
        """

        super().__init__()

        ################################################################
        # TODO
        self.out_channels = out_channels
        self.d = nn.MaxPool2d(kernel_size= 1, stride = stride)
        
        ################################################################

    def forward(self, x):

        """
        Applies spatial downsampling and zero-padding in the channel dimension.
        """

        ################################################################
        # TODO

        shortcut = self.d(x)
        in_channels = shortcut.shape[1]

        if in_channels < self.out_channels:
            pad_channels = self.out_channels - in_channels
            pad = shortcut.new_zeros(
                shortcut.shape[0],
                pad_channels,
                shortcut.shape[2],
                shortcut.shape[3],
            )

            shortcut = torch.cat((shortcut, pad), dim = 1)
        ################################################################

        return shortcut




class ProjectionShortcut(nn.Module):

    """
    Learned projection shortcut using a 1x1 convolution from option B.
    """

    def __init__(self, in_channels, out_channels, stride):

        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Spatial stride used by the residual branch.
        """

        super().__init__()

        ################################################################
        # TODO

        self.down = nn.Conv2d(in_channels, out_channels, 1, stride)

        ################################################################

    def forward(self, x):

        """
        Applies the learned projection shortcut.
        """

        ################################################################
        # TODO

        shortcut = self.down(x)

        ################################################################

        return shortcut
