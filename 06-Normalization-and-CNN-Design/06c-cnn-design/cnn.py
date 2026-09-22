import warnings
import torch
import torchvision
import torchvision.transforms.v2 as transforms
from torch import nn


class SimpleCNN(nn.Module):

    def __init__(self, num_classes = 10):

        """
        Creates a convolutional neural network for CIFAR-10 classification.

        Args:
            num_classes (int): Number of output classes.
        """

        super().__init__()

        ################################################################
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.05),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.05),

            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout2d(0.1),

            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout2d(0.1),

            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.2),

            nn.Flatten(),
            nn.Linear(128 * 4 *4, num_classes)
        )

        
        ################################################################

    def forward(self, x):

        """
        Calculates logits of the neural network for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32).

        Returns:
            logits (torch.Tensor): Tensor of raw output logits with shape
                (batch_size, num_classes).
        """

        ################################################################
        # TODO
        
        logits = self.features(x)
        
        ################################################################

        return logits

    def predict_class(self, x):
        
        """
        Predicts the class labels for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32).

        Returns:
            predicted_class (torch.Tensor): Tensor of predicted class indices
                with shape (batch_size,).
        """

        ################################################################
        # TODO
        
        logits = self(x)
        predicted_class = torch.argmax(logits, dim=-1)
        
        ################################################################

        return predicted_class


def initialize(lr, weight_decay, device):
    
    """
    Initializes the model and the AdamW optimizer.

    Args:
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay used by the optimizer.
        device (torch.device): The device on which the model lives.
    
    Returns:
        cnn (SimpleCNN): The initialized CNN model.
        optimizer (torch.optim.AdamW): The initialized AdamW optimizer.
    """

    ################################################################
    # TODO
    
    cnn = SimpleCNN().to(device)
    optimizer = torch.optim.AdamW(cnn.parameters(), lr = lr, weight_decay = weight_decay)
    ################################################################

    return cnn, optimizer


def training_step(cnn, optimizer, loss_fn, x, y, device):
    
    """
    Performs a single training step: forward pass, loss calculation, backward
    pass, and optimizer step.

    Args:
        cnn (SimpleCNN): The CNN model to train.
        optimizer (torch.optim.Optimizer): The optimizer to update the model parameters.
        loss_fn (torch.nn.Module): The loss function to calculate the training loss.
        x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32).
        y (torch.Tensor): Target tensor of shape (batch_size,).
        device (torch.device): The device on which the model lives.

    Returns:
        loss_value (float): The value of the training loss for the given batch.
    """
    
    ################################################################
    # TODO

    cnn.train()

    x = x.to(device)
    y = y.to(device)

    logits = cnn(x)
    loss = loss_fn(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_value = loss.item()
    

    ################################################################

    return loss_value








































try:
    import tqdm
except ModuleNotFoundError:
    class _SimpleTqdm:
        def __init__(self, iterable, desc=None):
            self.iterable = iterable
            self.desc = desc

        def __iter__(self):
            return iter(self.iterable)

        def set_description(self, desc):
            self.desc = desc

    class tqdm:
        @staticmethod
        def tqdm(iterable, desc=None):
            return _SimpleTqdm(iterable, desc=desc)


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def train(lr, batch_size, weight_decay, epochs = 5):
    """
    Trains the CNN on the CIFAR-10 dataset and tracks the train loss and test
    accuracy during training.

    Args:
        lr (float): Learning rate for the optimizer.
        batch_size (int): Number of samples in each training batch.
        weight_decay (float): Weight decay used by the optimizer.
        epochs (int): Number of epochs to train for.

    Returns:
        cnn (SimpleCNN): The trained CNN model.
        train_loss_history (list): List of training loss values for each epoch.
        test_acc_history (list): List of test accuracy values for each epoch.
    """
    # data preparation
    transform_f = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".*align should be passed as Python or NumPy boolean.*", category=Warning)

        train_data = torchvision.datasets.CIFAR10(
            root="./data", train=True, transform=transform_f, download=True)

        test_data = torchvision.datasets.CIFAR10(
            root="./data", train=False, transform=transform_f, download=True)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=0)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cnn, optimizer = initialize(lr, weight_decay, device)

    loss_fn = nn.CrossEntropyLoss()

    train_loss_history = []
    test_acc_history = []

    pbar = tqdm.tqdm(
        range(epochs),
        desc="Train loss: -.---, Test accuracy: -.---",
    )

    for _ in pbar:
        cnn.train()
        epoch_loss = 0.0

        for x, y in train_loader:
            loss_value = training_step(cnn, optimizer, loss_fn, x, y, device)
            epoch_loss += loss_value

        epoch_loss /= len(train_loader)
        train_loss_history.append(epoch_loss)

        cnn.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)

                predicted_class = cnn.predict_class(x)
                correct += (predicted_class == y).sum().item()
                total += y.numel()

        test_acc = correct / total
        test_acc_history.append(test_acc)

        pbar.set_description(
            f"Train loss: {epoch_loss:.3f}, Test accuracy: {100 * test_acc:.2f} %"
        )

    final_acc = test_acc_history[-1]
    print(f"\nFinal test accuracy: {100 * final_acc:.2f} %")

    return cnn, train_loss_history, test_acc_history
