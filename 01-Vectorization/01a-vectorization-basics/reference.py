import torch


def quick_sort_py_native(arr: list[float] ):
    """
    Sorts a python native list of numbers using quick sort.
    Args:
        arr: list of numbers

    Returns:
        sorted list of numbers
    """
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    less = []
    equal = []
    greater = []
    for element in arr:
        if element < pivot:
            less.append(element)
        elif element == pivot:
            equal.append(element)
        else:
            greater.append(element)
    return quick_sort_py_native(less) + equal + quick_sort_py_native(greater)

def quick_sort_naive(arr: torch.Tensor ):
    """
    Sort the elements of a tensor with quicksort, but is not vectorized.
    Args:
        arr: Tensor of shape (N, )

    Returns:
        sorted tensor of shape (N, )
    """
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    less = []
    equal = []
    greater = []
    for element in arr:
        if element < pivot:
            less.append(element.view(1))
        elif element == pivot:
            equal.append(element.view(1))
        else:
            greater.append(element.view(1))
    less = torch.cat(less) if len(less) > 0 else torch.empty(0)
    equal = torch.cat(equal) if len(equal) > 0 else torch.empty(0)
    greater = torch.cat(greater) if len(greater) > 0 else torch.empty(0)
    return torch.cat([quick_sort_naive(less), equal, quick_sort_naive(greater)])

def linear_prediction(X: torch.Tensor, w:torch.Tensor, b:torch.Tensor):
    """
    Calculates a linear prediction, but is not vectorized.
    Args:
        X: Data tensor of shape (N, D
        w: weight tensor of shape (D, )
        b: bias tensor of shape (, )

    Returns:
        Output tensor of shape (N, )
    """
    y = torch.empty(X.shape[0])
    for i in range(X.shape[0]):
        y[i] = torch.dot(w,X[i]) + b
    return y

def linear_prediction_native(X: list[list[float]], w:list[float], b:float):
    """
        Calculates a linear prediction with pthon native lists.
        Args:
            X: Nested list of shape (N, D)
            w: weight list of len D
            b: bias

        Returns:
            Output list of len N
    """
    y = []
    for x in X:  # Iterate over rows of X (samples)
        element = 0.0
        for i in range(len(x)):  # Iterate over features
            element += x[i] * w[i]
        element += b
        y.append(element)
    return y

def calculate_distances(X: torch.Tensor):
    """
    Calculates the pairwise distances between vectors, but is not vectorized.

    Args:
        X: Data tensor of shape (N, D)

    Returns:
        Distance matrix of shape (N, N)
    """
    N, D = X.shape
    dist = torch.zeros(N, N)
    for i in range(N):
        for j in range(i, N):
            diff = X[i, :] - X[j, :]
            dist[i, j] = diff.pow_(2).sum(dim=0)
            dist[i, j].sqrt_()

    dist = dist + dist.T
    return dist

def calculate_distance_identity(X: torch.Tensor):
    """
        Calculates the pairwise distances between vectors, via an identity rule using a binomial formula.

        Args:
            X: Data tensor of shape (N, D)

        Returns:
            Distance matrix of shape (N, N)
        """
    norm = torch.sum(X.pow(2), dim=1, keepdim=True)
    G = norm + norm.T - 2 * X @ X.T
    G.fill_diagonal_(0) # due to numeric instability, this can be non-zero.
    G.sqrt_()
    return G