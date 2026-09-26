import torch
from torch import nn


class CNNEncoder(nn.Module):
    """
    Small CNN encoder for CIFAR-10 images.

    The encoder maps an input image to a feature vector of size `out_dim`.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 128

    def forward(self, x):
        """
        Computes image features.

        Args:
            x (Tensor): Image batch of shape (B, 3, 32, 32).

        Returns:
            features (Tensor): Feature tensor of shape (B, out_dim).
        """
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x


class RotationClassifier(nn.Module):
    """
    Classifier for the RotNet pretraining task.

    The model predicts whether an input image was rotated by 0, 90, 180,
    or 270 degrees.
    """

    def __init__(self):
        super().__init__()

        ################################################################
        # TODO

        self.encoder = CNNEncoder()
        self.head = nn.Sequential(
            nn.Linear(self.encoder.out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )
        
        
        ################################################################

    def forward(self, x):
        """
        Predicts rotation logits for a batch of images.

        Args:
            x (Tensor): Image batch of shape (B, 3, 32, 32).

        Returns:
            logits (Tensor): Rotation logits of shape (B, 4).
        """

        ################################################################
        # TODO

        return self.head(self.encoder(x))
        
        
        ################################################################

    def predict_class(self, x):
        """
        Predicts the class labels for the given input tensor.

        Args:
            x (Tensor): Input tensor of shape (B, 3, 32, 32).

        Returns:
            predicted_class (Tensor): Tensor of predicted class indices
                with shape (B,).
        """
        logits = self.forward(x)
        predicted_class = torch.argmax(logits, dim=1)
        return predicted_class


class DownstreamClassifier(nn.Module):
    """
    Classifier for the downstream CIFAR-10 task.

    The encoder is used as a fixed feature extractor. Only the classification
    head is trained.
    """

    def __init__(self, encoder):
        """
        Initializes the downstream classifier.

        Args:
            encoder (CNNEncoder): Pretrained encoder from the rotation classifier.
        """
        super().__init__()

        ################################################################
        # TODO
        self.encoder = encoder
        self.head = nn.Linear(self.encoder.out_dim, 10)
        ################################################################

    def forward(self, x):
        """
        Predicts CIFAR-10 logits for a batch of images.

        Args:
            x (Tensor): Image batch of shape (B, 3, 32, 32).

        Returns:
            logits (Tensor): CIFAR-10 logits of shape (B, 10).
        """

        ################################################################
        # TODO
        self.encoder.eval()
        with torch.no_grad():
            x = self.encoder(x)
        return self.head(x)

        ################################################################

    def predict_class(self, x):
        """
        Predicts the class labels for the given input tensor.

        Args:
            x (Tensor): Input tensor of shape (B, 3, 32, 32).

        Returns:
            predicted_class (Tensor): Tensor of predicted class indices
                with shape (B,).
        """
        logits = self.forward(x)
        predicted_class = torch.argmax(logits, dim=1)
        return predicted_class
