import torch


def kmeans_plusplus(X: torch.Tensor, k:int):
    """
    Initializes cluster centers via k-means++ algorithm
    Args:
        X: Data tensor of shape (N, D)
        k: Number of cluster centers

    Returns:
        Tensor of shape (k, D)
    """
    selected = torch.zeros(len(X), dtype=torch.bool)
    first = torch.randint(0, len(X), (1,))
    selected[first] = True

    for i in range(1, k):
        dist = torch.cdist(X[~selected], X[selected]) # N-k x k
        min_dist, _ = torch.min(dist, dim=1) # N-k
        idx = torch.multinomial(min_dist, 1)
        tmp = selected[~selected]
        tmp[idx] = True
        selected[~selected] = tmp
    return X[selected]

def k_means(X: torch.Tensor, k: int):
    """
        Cluster data using k-means algorithm
        Args:
            X: Data tensor of shape (N, D)
            k: Number of clusters

        Returns:
            Tuple (assignments, centers, i)
            assignments: Cluster assignments tensor of shape (N, )
            centers: Cluster center tensor of shape (k, D)
            i: number of iterations the algorithm needed to converge
        """
    N, D = X.shape
    centers = kmeans_plusplus(X, k)  # k x D

    # calculate initial cluster assignments
    assignments = torch.zeros(N, dtype=torch.long)
    for n in range(N):
        min_dist = torch.inf
        min_idx = -1
        for j in range(k):
            dist = torch.sqrt(torch.sum((X[n] - centers[j]) ** 2))
            if dist < min_dist:
                min_dist = dist
                min_idx = j

        assignments[n] = min_idx

    for i in range(100): # Limit the maximum number of loop iterations to avoid livelock
        # M step: update the cluster centers to be the mean of its elements
        for j in range(k):
            sum = torch.zeros(D)
            num_elements = 0
            for n in range(N):
                if assignments[n] == j:
                    sum += X[n]
                    num_elements += 1
            # only update centers that have elements (div by 0!), otherwise ignore
            if num_elements > 0:
                centers[j] = sum / num_elements

        # E step: calculate which center has the lowest distance to all data points
        new_assignments = torch.empty_like(assignments)
        for n in range(N):
            min_dist = torch.inf
            min_idx = -1
            for j in range(k):
                dist = torch.sqrt(torch.sum((X[n] - centers[j]) ** 2))
                if dist < min_dist:
                    min_dist = dist
                    min_idx = j

            new_assignments[n] = min_idx

        # check for convergance
        if torch.all(assignments == new_assignments):
            break
        assignments = new_assignments

    return assignments, centers, i