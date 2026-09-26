import copy

import torch
from torch import nn

################################################################
# TODO: Choose and implement an encoder architecture.
################################################################


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
        self.out_channels = out_channels
        self.downsample = nn.MaxPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        """
        Applies spatial downsampling and zero-padding in the channel dimension.
        """

        shortcut = self.downsample(x)
        in_channels = shortcut.shape[1]

        if self.out_channels > in_channels:
            pad_channels = self.out_channels - in_channels
            padding = shortcut.new_zeros(
                shortcut.shape[0],
                pad_channels,
                shortcut.shape[2],
                shortcut.shape[3],
            )
            shortcut = torch.cat([shortcut, padding], dim=1)

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
        self.projection = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False, # True is ok, but BatchNorm2d cancels it out anyway.
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        """
        Applies the learned projection shortcut.
        """
        shortcut = self.projection(x)
        return shortcut


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

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Identity()
        elif shortcut == "projection":
            self.shortcut = ProjectionShortcut(
                in_channels,
                out_channels,
                stride,
            )
        elif shortcut == "zero_pad":
            self.shortcut = ZeroPadShortcut(out_channels, stride)
        else:
            raise ValueError(
                "shortcut must be either 'zero_pad' or 'projection'"
            )

    def forward(self, x):
        """
        Applies the residual block.
        """

        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        shortcut = self.shortcut(x)
        out = self.relu(residual + shortcut)
        return out


def make_block_group(in_channels, out_channels, num_blocks, stride, shortcut):
    """
    Creates a group of residual blocks with the same output shape.
    """

    blocks = [
        BasicBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            shortcut=shortcut,
        )
    ]

    for _ in range(1, num_blocks):
        blocks.append(
            BasicBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
                shortcut=shortcut,
            )
        )

    block_group = nn.Sequential(*blocks)
    return block_group


class ResNet(nn.Module):

    """
    CIFAR-style ResNet.
    """

    def __init__(self, num_blocks, channels, shortcut):
        """
        Args:
            num_blocks (tuple[int, int, int]): Number of blocks in the three
                block groups.
            channels (tuple[int, int, int]): Channel counts for the block
                groups.
            out_dim (int): Output dimension for the final projection.
            shortcut (str): Shortcut type, "zero_pad" or "projection".
        """

        super().__init__()

        c1, c2, c3 = channels
        n1, n2, n3 = num_blocks

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                c1,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.blocks_32x32 = make_block_group(
            c1,
            c1,
            n1,
            stride=1,
            shortcut=shortcut,
        )
        self.blocks_16x16 = make_block_group(
            c1,
            c2,
            n2,
            stride=2,
            shortcut=shortcut,
        )
        self.blocks_8x8 = make_block_group(
            c2,
            c3,
            n3,
            stride=2,
            shortcut=shortcut,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = c3

    def forward(self, x):
        """
        Calculates raw class logits.
        """
        x = self.stem(x)
        x = self.blocks_32x32(x)
        x = self.blocks_16x16(x)
        x = self.blocks_8x8(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def resnet20():
    """
    Creates the CIFAR ResNet-20 configuration from the paper.
    """

    return ResNet(
        num_blocks=(3, 3, 3),
        channels=(32, 64, 128),
        shortcut="projection",
    )

################################################################
# End of freely choosable encoder block.
################################################################


class BYOL(nn.Module):
    """
    BYOL model with online and target networks.

    Required attributes:
        online_encoder
        online_projector
        online_predictor
        target_encoder
        target_projector
    """

    def __init__(
        self,
        projection_dim=256,
        projection_hidden_dim=512,
        prediction_hidden_dim=512,
    ):
        super().__init__()

        ################################################################
        # TODO

        self.online_encoder = resnet20()
        self.online_projector = nn.Sequential(
            nn.Linear(self.online_encoder.out_dim, projection_hidden_dim),
            nn.ReLU(),
            nn.Linear(projection_hidden_dim, projection_dim)
        )

        self.online_predictor = nn.Sequential(
            nn.Linear(projection_dim, prediction_hidden_dim),
            nn.ReLU(),
            nn.Linear(prediction_hidden_dim, projection_dim),
        )

        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)

        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        for p in self.target_projector.parameters():
            p.requires_grad_(False)
        ################################################################

    def forward(self, x):
        """
        Encodes images using the online encoder.
        """
        return self.online_encoder(x)

    def online_forward(self, x):
        """
        Returns online features, projections, and predictions.
        """

        ################################################################
        # TODO
        features = self.online_encoder(x)
        projections = self.online_projector(features)
        predictions = self.online_predictor(projections)
        
        ################################################################
        return features, projections, predictions

    def target_forward(self, x):
        """
        Returns target features and projections.
        """

        ################################################################
        # TODO
        
        features = self.target_encoder(x)
        projections = self.target_projector(features)

        
        ################################################################
        return features, projections


class LinearClassifier(nn.Module):
    """
    Linear classifier on top of a frozen encoder.
    """

    def __init__(self, encoder, num_classes=10):
        super().__init__()

        self.encoder = encoder
        self.head = nn.Linear(encoder.out_dim, num_classes)

    def forward(self, x):
        self.encoder.eval()
        with torch.no_grad():
            features = self.encoder(x)
        logits = self.head(features)
        return logits

    def predict_class(self, x):
        logits = self.forward(x)
        predicted_class = torch.argmax(logits, dim=1)
        return predicted_class
