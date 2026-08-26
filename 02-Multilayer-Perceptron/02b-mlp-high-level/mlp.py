import tqdm
import torch
import torchvision
from torch import nn
import torchvision.transforms.v2 as transforms

class SimpleMLP(nn.Module):

    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int):

        """
        Creates a two layer MLP with ReLU activation for classification tasks.

        Args:
            dim_in     (int): The input dimension.
            dim_hidden (int): The hidden layer dimension.
            dim_out    (int): The output dimension (number of classes).
        """
        super().__init__()

        ################################################################
        # TODO
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_out),
        )

        

        ################################################################

    def forward(self, x: torch.Tensor):

        """
        Calculates logits (raw output values) of the neural network for the given input tensor x

        Args:
            x (Tensor): Input tensor of shape (batch_size, dim_in)

        Returns:
            logits (Tensor): Tensor of raw output logits of the neural network with shape (batch_size, dim_out)
        """

        ################################################################
        # TODO
        
        logits = self.mlp(x)
        ################################################################

        return logits

    def predict_class(self, x: torch.Tensor):

        """
        Predicts the class labels for the given input tensor x.

        Args:
            x (Tensor): Input tensor of shape (batch_size, dim_in)

        Returns:
            class_prediction (Tensor): Tensor of predicted classes with shape (batch_size, ). Contains the predicted class labels (integers) for each input sample in the batch.
        """

        ################################################################
        # TODO
        
        logits = self.mlp(x)
        predicted_class = torch.argmax(logits, dim = -1)

        ################################################################

        return predicted_class

def train(lr, batch_size, epochs = 10):

    """
    Trains the MLP on the MNIST dataset and tracks the train error and test accuracy during training.

    Args:
        lr (float): learning rate for the optimizer
        batch_size (int): number of samples in each batch for training
        epochs (int): number of epochs to train the MLP

    Returns:
        mlp (SimpleMLP): the trained MLP model
        train_error_history (list): list of train error values for each epoch
        test_acc_history (list): list of test accuracy values for each epoch
    """

    # Data preparation
    transform_f = transforms.Compose([transforms.ToImage(), torch.flatten, transforms.ToDtype(torch.float)])
    train_data = torchvision.datasets.MNIST(root='./data', train=True, transform=transform_f, download=True)
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = torchvision.datasets.MNIST(root='./data', train=False, transform=transform_f)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    ################################################################
    # TODO initialize the mlp, the loss_fn and the optimizer
    
    mlp = SimpleMLP(dim_in = 784, dim_hidden = 128, dim_out = 10)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(mlp.parameters(), lr = lr)
    
    ################################################################

    train_error_history = []
    test_acc_history = []

    pbar = tqdm.tqdm(range(epochs), desc=f"Train error: -.---, Test accuracy: -.---")

    for epoch in pbar:
        epoch_loss = 0.0

        for x, y in dataloader:

            ################################################################
            # TODO
            yp = mlp(x)
            loss = loss_fn(yp, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ################################################################

            epoch_loss += loss.item()

        with torch.no_grad():
            # track train loss
            epoch_loss /= len(dataloader)
            train_error_history.append(epoch_loss)

            # track test accuracy
            test_acc = 0.0
            for x, y in test_loader:
                out = mlp.predict_class(x)
                test_acc += torch.mean(out == y, dtype=torch.float).item()
            test_acc /= len(test_loader)
            test_acc_history.append(test_acc)

        pbar.set_description(f"Train error: {epoch_loss:.3f}, Test accuracy: {100*test_acc:.3f} % correct")
        
    return mlp, train_error_history, test_acc_history