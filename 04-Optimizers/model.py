import torch.nn as nn


# Default experiment/model settings
POINTS_PER_CLASS = 80
INPUT_DIM = 32
HIDDEN_DIM = 128
OUTPUT_DIM = 3


class MLP(nn.Module):

    """
    Small MLP for synthetic classification.

    The focus of this exercise is the optimizer implementation,
    not the model architecture.
    """

    def __init__(
        self,
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
