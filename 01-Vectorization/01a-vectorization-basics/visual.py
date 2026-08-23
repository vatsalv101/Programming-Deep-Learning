import statistics
import plotly.express as px


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