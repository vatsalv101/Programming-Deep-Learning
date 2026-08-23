import torch
from reference import kmeans_plusplus

def k_means_vec(X: torch.Tensor, k:int):

    """
    Vectorized implementation of k-means algorithm

    Args:
        X (Tensor): Data tensor of shape (N, D)
        k (int): Number of clusters

    Returns:
        assignments (Tensor): Cluster assignments tensor of shape (N, )
        centers (Tensor): Cluster center tensor of shape (k, D)
        i (int): number of iterations the algorithm needed to converge
    """

    # Initialization
    centers = kmeans_plusplus(X, k)  # k x D

    ################################################################
    # TODO: calculate the initial cluster assignment
    
    dist = torch.sqrt(torch.sum((X.unsqueeze(0) - centers.unsqueeze(1))**2, dim = -1))
    assignments = torch.argmin(dist, dim = 0)
    
    ################################################################

    for i in range(100): # Limit the maximum number of loop iterations to avoid livelock

        # M step

        ################################################################
        # TODO: calculate the new centers
        for j in range(k):
            centers[j] = torch.mean(X[assignments == j], dim = -1)
        ################################################################

        # E step

        ################################################################
        # TODO calculate new_assignments
        dist = torch.sqrt(torch.sum((X.unsqueeze(0) - centers.unsqueeze(1))**2, dim = -1))
        new_assignments = torch.argmin(dist, dim = 0)
        ################################################################

        if torch.all(assignments == new_assignments):
            break
        assignments = new_assignments

    return assignments, centers, i

def k_means_scatter(X: torch.Tensor, k: int):

    """
    Improved vectorized implementation of k-means algorithm using scatter_reduce.

    Args:
        X (Tensor): Data tensor of shape (N, D)
        k (int): Number of clusters

    Returns:
        assignments (Tensor): Cluster assignments tensor of shape (N, )
        centers (Tensor): Cluster center tensor of shape (k, D)
        i (int): number of iterations the algorithm needed to converge
    """

    N, D = X.shape

    centers = kmeans_plusplus(X, k)  # k x D

    dist = torch.cdist(centers, X)  # k x N
    assignments = torch.argmin(dist, dim=0)

    for i in range(100): # Limit the maximum number of loop iterations to avoid livelock

        # M step

        ################################################################
        # TODO: calculate the new centers with scatter_reduce_()

        index = assignments.unsqueeze(-1).expand(N, D)
        centers = centers.scatter_reduce_(0, index, X, reduce='mean', include_self=False)
        ################################################################

        # E step
        
        dist = torch.cdist(centers, X)  # k x N
        new_assignments = torch.argmin(dist, dim=0, keepdim=False)  # 1 x N

        if torch.all(assignments == new_assignments):
            break
        assignments = new_assignments

    return assignments, centers, i

