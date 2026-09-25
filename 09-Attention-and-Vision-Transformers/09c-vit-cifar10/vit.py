import torch
import torch.optim as optim
from torch import nn
from encoder import create_transformer_encoder
from patchify import create_patch_sequence
import warnings
import torchvision
import torchvision.transforms.v2 as transforms
import data

class ViT(nn.Module):

    def __init__(self, num_layers=9, model_dim=192, num_heads=12, patch_size=4, num_classes=10):

        """
        Creates a Vision Transformer (ViT) for CIFAR-10 classification.

        Args:
            num_layers (int): Number of Transformer encoder layers.
            model_dim (int): Dimension of the input and output features.
            num_heads (int): Number of attention heads.
            patch_size (int): Horizontal and vertical size of each image patch.
            num_classes (int): Number of output classes.
        """

        super().__init__()

        ################################################################
        # TODO
        self.patch_size = patch_size
        num_patches = (32 // patch_size) ** 2
        self.patch_proj = nn.Linear(3 * patch_size * patch_size, model_dim)

        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.Tanh(),
            nn.Linear(model_dim, num_classes),
        )

        self.cls = nn.Parameter(torch.randn(1, 1, model_dim))
        self.pos = nn.Parameter(torch.randn(1, num_patches + 1, model_dim))

        self.encoder = create_transformer_encoder(num_layers=num_layers, model_dim=model_dim, num_heads=num_heads, mlp_hidden_dim = 4 * model_dim)

        
        ################################################################

    def forward(self, x):

        """
        Forward pass of the ViT model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """

        ################################################################
        # TODO
        seq = create_patch_sequence(x, self.patch_size)
        seq = self.patch_proj(seq)

        clst = self.cls.expand(x.shape[0], -1, -1)
        z0 = torch.cat((clst, seq), dim = 1)
        z0 = z0 + self.pos

        z1 = self.encoder(z0)

        out = z1[:, 0, :]

        return self.mlp(out)
         

        ################################################################

    def predict_class(self, x):

        """
        Predicts the class labels for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32).

        Returns:
            predicted_class (torch.Tensor): Tensor of predicted class indices
                with shape (batch_size,).
        """

        predicted_class = None
    
        logits = self.forward(x)
        predicted_class = torch.argmax(logits, dim=1)

        return predicted_class
    

def training_step(vit, optimizer, loss_fn, x, y, device):
    
    """
    Performs a single training step: forward pass, loss calculation, backward
    pass, and optimizer step.

    Args:
        vit (ViT): The ViT model to train.
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

    x = x.to(device)
    y = y.to(device)

    logits = vit(x)
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


def train(lr, batch_size, weight_decay, epochs):

    """
    Trains the ViT on the CIFAR-10 dataset and tracks the train loss and test
    accuracy during training.

    Args:
        lr (float): Learning rate for the optimizer.
        batch_size (int): Number of samples in each training batch.
        weight_decay (float): Weight decay used by the optimizer.
        epochs (int): Number of epochs to train for.

    Returns:
        model (ViT): The trained ViT model.
        train_loss_history (list): List of training loss values for each epoch.
        test_acc_history (list): List of test accuracy values for each epoch.
    """

    #############################################################
    # TODO

    train_loader, test_loader = data.make_cifar10_loaders(batch_size)

    model = ViT()
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs)
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device('cpu')
    train_loss_history = []    
    test_acc_history = []

    model.train()
    train_loss = 0
    train_samples = 0
    for x, y in train_loader:
        bs = x.shape[0]
        train_loss += training_step(model, optimizer, loss_fn, x, y, device) * bs
        train_samples += bs
    train_loss_history.append(train_loss / train_samples)
    scheduler.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            yp = model.predict_class(x)
            correct += (yp == y).sum()
            total += y.numel()

    test_acc_history.append(correct / total)    
    
    #############################################################

    return model, train_loss_history, test_acc_history