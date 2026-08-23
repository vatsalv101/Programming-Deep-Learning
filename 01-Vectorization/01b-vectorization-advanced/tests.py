import timeit
import torch
import sys

import reference
import clustering
import visual


def check_cluster_valid(X, c, a):
    dist = torch.cdist(X, c)
    assigned_dist = dist.gather(dim=1, index=a.unsqueeze(-1)).squeeze()
    min_dist, _ = torch.min(dist, dim=1)
    valid = True

    if not torch.allclose(assigned_dist, min_dist):
        diff = assigned_dist - min_dist
        valid = False
        print(f"Not the smallest distance to assigned cluster centers for data points at index {torch.where(torch.abs(diff) > 1e-8)[0].tolist()}", file=sys.stderr)

    for i in range(len(c)):
        assumed_center = torch.mean(X[a == i], dim=0)
        if  not torch.allclose(assumed_center, c[i], atol=1e-4):
            valid = False
            print(f"Center {i} should be {assumed_center} but is {c[i]}", file=sys.stderr)
            print(f"Diff: {assumed_center - c[i]}", file=sys.stderr)
    if valid:
        print("The clusters are a valid solution")
    else:
        print("The clusters are NOT a valid solution", file=sys.stderr)

def sample_data(N, num_generators = 7, D=2, hypersphere = False):
    alphas = torch.rand(num_generators)
    means = 15 * (torch.rand((num_generators, D)) - 0.5)
    sigmas = torch.randn(num_generators, D, D)
    sigmas = sigmas @ sigmas.transpose(1, 2)

    dist = torch.distributions.MixtureSameFamily(mixture_distribution=torch.distributions.Categorical(probs=alphas),
                                                 component_distribution=torch.distributions.MultivariateNormal(
                                                     loc=means, covariance_matrix=sigmas))
    data = dist.sample((N,))
    if hypersphere:
        data /= data.norm(dim=-1, keepdim=True)
    return data

def benchmark_kmeans():
    N = 200
    D = 2
    K = 5

    num = 5
    repeat = 5
    naive_times = timeit.repeat(lambda: reference.k_means(torch.randn(N, D), K), number=num, repeat=repeat)
    vec_times = timeit.repeat(lambda: clustering.k_means_vec(torch.randn(N, D), K), number=num, repeat=repeat)
    return visual.plot_comparison({"naive": naive_times, "vectorized": vec_times}, "K-Means: Vectorization Speedup")

def benchmark_kmeans_scatter():
    N = 5_000
    D = 6
    K = 25

    num = 5
    repeat = 5
    # naive_times = timeit.repeat(lambda: vec.k_means(randn(N, D), K), number=5, repeat=10) # this would take too long :(
    vec_times = timeit.repeat(lambda: clustering.k_means_vec(torch.randn(N, D), K), number=num, repeat=repeat)
    scatter_time = timeit.repeat(lambda: clustering.k_means_scatter(torch.randn(N, D), K), number=num, repeat=repeat)
    return visual.plot_comparison({"vector": vec_times, "scatter": scatter_time}, "K-Means")