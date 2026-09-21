import plotly.graph_objects as go
from plotly.subplots import make_subplots


def show_training_comparison(histories):
    """
    Plots training loss and test accuracy for multiple models.

    Args:
        histories (dict): Dictionary returned by cnn.train_short_comparison.

    Returns:
        plotly.graph_objects.Figure: Figure with training curves.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Training Loss", "Test Accuracy"],
    )

    for name, history in histories.items():
        epochs = list(range(1, len(history["train_loss"]) + 1))

        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history["train_loss"],
                mode="lines+markers",
                name=f"{name} loss",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history["test_acc"],
                mode="lines+markers",
                name=f"{name} accuracy",
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(title_text="epoch", row=1, col=1)
    fig.update_xaxes(title_text="epoch", row=1, col=2)
    fig.update_yaxes(title_text="loss", row=1, col=1)
    fig.update_yaxes(title_text="accuracy", row=1, col=2)
    fig.update_layout(title="Short CNN Comparison")

    return fig
