import matplotlib.pyplot as plt
import torch

import weight_init


def _simulate_mlp(
    init_fn_,
    activation_fn,
    depth=100,
    width=512,
    batch_size=1024,
    input_std=1.0,
    seed=0,
):
    """
    Simulates forward propagation through a deep MLP.

    Returns:
        activation_stds: list of activation standard deviations at each layer
    """
    torch.manual_seed(seed)

    x = torch.randn(batch_size, width) * input_std
    activation_stds = [x.std()]

    for layer in range(depth):
        weights = torch.empty(width, width)
        init_fn_(weights, n_in=width, n_out=width)

        x = x @ weights
        x = activation_fn(x)

        activation_stds.append(x.std())

    return activation_stds


def visualize_activation_stds(depth=30, width=512, batch_size=1024, input_std=1.0, constant=0.005):
    init_fns_ = [
        ("Constant + Sigmoid", lambda w, n_in, n_out: w.fill_(constant), torch.sigmoid, "C0", "-"),
        ("Constant + ReLU", lambda w, n_in, n_out: w.fill_(constant), torch.relu, "C0", "--"),
        ("LeCun + Sigmoid", weight_init.lecun_init_, torch.sigmoid, "C1", "-"),
        ("LeCun + ReLU", weight_init.lecun_init_, torch.relu, "C1", "--"),
        ("Xavier / Glorot + Sigmoid", weight_init.xavier_init_, torch.sigmoid, "C2", "-"),
        ("Xavier / Glorot + ReLU", weight_init.xavier_init_, torch.relu, "C2", "--"),
        ("Kaiming He + Sigmoid", weight_init.kaiming_he_init_, torch.sigmoid, "C3", "-"),
        ("Kaiming He + ReLU", weight_init.kaiming_he_init_, torch.relu, "C3", "--"),
    ]

    plt.figure(figsize=(8, 5))

    for name, init_fn_, activation_fn, color, linestyle in init_fns_:
        stds = _simulate_mlp(
            init_fn_=init_fn_,
            activation_fn=activation_fn,
            depth=depth,
            width=width,
            batch_size=batch_size,
            input_std=input_std,
            seed=0,
        )
        plt.plot(range(depth + 1), stds, label=name, color=color, linestyle=linestyle)

    plt.xlabel("MLP depth / layer")
    plt.ylabel("Standard deviation of activations")
    plt.title(f"Activation std vs depth for different initializations")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")
    plt.tight_layout()
    plt.show()
