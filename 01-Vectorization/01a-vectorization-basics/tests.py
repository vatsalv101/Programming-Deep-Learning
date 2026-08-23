import sys
from typing import Callable
import torch
import visual
import timeit
import reference
import vec

def test_sort(sort_fn: Callable[[torch.Tensor], torch.Tensor]):
    data_in = torch.randn(100)
    data_out = sort_fn(data_in)
    check_sorted(data_out)

def check_sorted(arr: torch.Tensor):
    greater =  arr[1:] >= arr[:-1]
    if torch.all(greater):
        print("Sorted")
    else:
        idx = torch.where(~greater)[0]
        for i in idx:
            print(f"Error: Element at index {i} is greater than {i+1}: {arr[i]}:.3f > {arr[i+1]}:.3f.", file=sys.stderr)

def test_linear(linear_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]):
    X = torch.randn(5, 3)
    w = torch.randn(3)
    b = torch.randn(1)
    ref = reference.linear_prediction(X, w, b)
    predicted = linear_fn(X, w, b)
    close = torch.isclose(ref, predicted)

    if torch.all(close):
        print("The prediction is correct")
    else:
        idx = torch.where(~close)[0]
        for i in idx:
            print(f"Error: Prediction at index {i} is wrong, got {predicted[i]:.3f} instead of {ref[i]:.3f}.", file=sys.stderr)

def test_distance(distance_fn: Callable[[torch.Tensor], torch.Tensor]):
    N = 10
    D = 15

    X = torch.randn(N, D)
    distances = reference.calculate_distances(X)
    distances_vec = distance_fn(X)

    close = torch.isclose(distances, distances_vec)

    if torch.all(close):
        print("The distances are correct")
    else:
        idx = torch.argwhere(~close)
        for i in idx:
            row, col = i[0], i[1]
            print(f"Error: Distance at index {row}, {col} is wrong, got {distances_vec[row, col]} instead of {distances[row, col]}", file=sys.stderr)


def benchmark_sort():
    N = 1_000

    naive_times = timeit.repeat(lambda: reference.quick_sort_naive(torch.randn(N)), number=5, repeat=10)
    vec_times = timeit.repeat(lambda: vec.quick_sort_vec(torch.randn(N)), number=5, repeat=10)
    py_native_times = timeit.repeat(lambda: reference.quick_sort_py_native(torch.randn(N).tolist()), number=5, repeat=10)
    torch_sort_times = timeit.repeat(lambda: torch.sort(torch.randn(N)), number=5, repeat=10)
    return visual.plot_comparison(
        {"naive": naive_times, "vectorized": vec_times, "python native": py_native_times, "sort": torch_sort_times},
        "Quicksort")

def benchmark_sum():
    def loop_sum(tensor):
        sum = 0
        for x in tensor:
            sum += x
        return sum

    N = 1_000
    for_loop_sum = timeit.repeat(lambda: loop_sum(torch.randn(N)), number=10, repeat=10)
    torch_sum = timeit.repeat(lambda: torch.sum(torch.randn(N)), number=10, repeat=10)

    return visual.plot_comparison({"loop sum": for_loop_sum, "torch.sum": torch_sum}, "Summing")

def benchmark_linear():
    N = 10_000
    D = 150

    naive_times = timeit.repeat(lambda: reference.linear_prediction(torch.randn(N, D), torch.randn(D), b=torch.randn(1)), number=5, repeat=10)
    vec_times = timeit.repeat(lambda: vec.linear_prediction_vec(torch.randn(N, D), torch.randn(D), b=torch.randn(1)), number=5, repeat=10)
    native_times = timeit.repeat(
        lambda: reference.linear_prediction_native(torch.randn(N, D).tolist(), torch.randn(D).tolist(), b=torch.randn(1).item()), number=5,
        repeat=10)
    return visual.plot_comparison(
        {"pytorch loop": naive_times, "python native": native_times, "pytorch vectorized": vec_times}, "Linear Model")

def benchmark_distance():
    N = 100
    D = 25

    def torch_dist(X):
        return torch.cdist(X, X)

    naive_times = timeit.repeat(lambda: reference.calculate_distances(torch.randn(N, D)), number=5, repeat=20)
    vec_times = timeit.repeat(lambda: vec.calculate_distance_vec(torch.randn(N, D)), number=5, repeat=20)
    id_times = timeit.repeat(lambda: reference.calculate_distance_identity(torch.randn(N, D)), number=5, repeat=20)
    cdist_times = timeit.repeat(lambda: torch_dist(torch.randn(N, D)), number=5, repeat=20)
    return visual.plot_comparison({"loop": naive_times, "vectorized": vec_times, "identity": id_times, "cdist": cdist_times},
                           "Pairwise Distance Calculation")