import plotly.express as px
import torchvision
import torchvision.transforms.v2 as transforms
import torch
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from mlp import SimpleMLP


def mnist_pictures(n = 2):
    transform_f = transforms.Compose([transforms.ToImage(), torch.flatten, transforms.ToDtype(torch.float)])
    test_data = torchvision.datasets.MNIST(root='./data', train=False, transform=transform_f, download=True)

    idx = torch.randperm(len(test_data))[:n]
    imgs = []
    labels = []

    for i in idx:
        imgs.append(test_data[i][0].reshape(1, 28, 28))
        labels.append(test_data[i][1])
    imgs = torch.cat(imgs)


    fig = px.imshow(imgs, binary_string=True, facet_col= 0, labels={'facet_col' :'labels'}, title="MNIST Predictions")

    for i, label in enumerate(labels):
        fig.layout.annotations[i]['text'] = f"label: {label}"

    return fig

def mnist_predictions(model: SimpleMLP, n = 2):
    transform_f = transforms.Compose([transforms.ToImage(), torch.flatten, transforms.ToDtype(torch.float)])
    test_data = torchvision.datasets.MNIST(root='./data', train=False, transform=transform_f, download=True)

    idx = torch.randperm(len(test_data))[:n]
    imgs = []
    labels = []

    for i in idx:
        imgs.append(test_data[i][0].reshape(1, -1))
        labels.append(test_data[i][1])
    imgs = torch.cat(imgs)
    predicted = model.predict_class(imgs)


    fig = px.imshow(imgs.reshape(-1, 28, 28), binary_string=True, facet_col= 0, labels={'facet_col' :'labels'}, title="MNIST Predictions")

    for i, label in enumerate(labels):
        fig.layout.annotations[i]['text'] = f"prediction: {predicted[i]}, label: {label}"

    return fig

def show_training_stats(train_loss, test_acc):
    fig = make_subplots(rows=2, cols=1, subplot_titles=["Training Loss", "Test Accuracy"])
    fig.add_trace(go.Scatter(x=torch.arange(len(train_loss)), y=train_loss), row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.add_trace(go.Scatter(x=torch.arange(len(test_acc)), y=test_acc), row=2, col=1)
    fig.update_layout(showlegend=False, title="MLP Training")
    return fig
