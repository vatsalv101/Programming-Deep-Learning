import statistics

import plotly.express as px
import plotly.graph_objects as go
import torch

def plot_comparison(data: dict[str, list[float]], title: str = "Comparison", axis=["method", "time [s]"]):
    medians = []
    maximum = []
    minimum = []

    for d in data.values():
        median = statistics.median(d)
        medians.append(median)
        maximum.append(max(d) - median)
        minimum.append(median - min(d))

    fig = px.bar(x=data.keys(), y=medians, error_y=maximum, error_y_minus=minimum, title=title, log_y=True)
    fig.update_layout(
        xaxis=dict(
            title=dict(
                text=axis[0]
            )
        ),
        yaxis=dict(
            title=dict(
                text=axis[1]
            )
        ),
    )
    return fig

def plot_clusters(X: torch.Tensor, c: torch.Tensor, a=None, title: str = None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X[:, 0,], y=X[:, 1],
                             mode="markers", marker_color=a))
    fig.add_trace(go.Scatter(x=c[:, 0],
                             y=c[:, 1],
                             mode="markers",
                             name="cluster centers",
                             marker_symbol="x",
                             marker_size=10,
                             marker_line_width=2,
                             marker_line_color="gray",
                             marker_color=torch.arange(len(c))))
    fig.update_traces(showlegend=False)
    fig.update(layout_coloraxis_showscale=False)
    fig.layout.yaxis.scaleanchor="x"
    fig.update_layout(title_text=title, height=600)
    return fig

def plot_clusters_3d(X: torch.Tensor, c: torch.Tensor, a=None, title: str = None):
    fig = px.scatter_3d(x=X[:, 0,], y=X[:, 1], z=X[:, 2], color=a, title=title)

    fig.add_trace(go.Scatter3d(x=c[:, 0],
                               y=c[:, 1],
                               z=c[:, 2],
                               mode="markers",
                               name="cluster centers",
                               marker_symbol="x",
                               marker_size=8,
                               marker_line_color="gray",
                               marker_line_width=2,
                               marker_color=torch.arange(len(c))))

    #fig.update_traces(showlegend=False)
    fig.update(layout_coloraxis_showscale=False)
    #fig.layout.yaxis.scaleanchor="x"
    return fig