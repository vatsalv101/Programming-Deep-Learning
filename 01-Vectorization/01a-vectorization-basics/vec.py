import torch

def quick_sort_vec(arr: torch.Tensor ):

    """
    Sorts the elements of a tensor using the quick sort algorithm.

    Args:
        arr (Tensor): unsorted input tensor of shape (N, )

    Returns:
        result (Tensor): sorted tensor of shape (N, )
    """

    if len(arr) <= 1:
        return arr

    ################################################################
    # TODO
    
    pivot = arr[len(arr) // 2]
    less = arr[arr < pivot]
    equal = arr[arr == pivot]
    greater = arr[arr > pivot]
    return torch.cat([quick_sort_vec(less), quick_sort_vec(equal), quick_sort_vec(greater)])
    ################################################################

    return result

def linear_prediction_vec(X: torch.Tensor, w:torch.Tensor, b:torch.Tensor):

    """
    Calculates the linear prediction for the given input data using weights and bias.

    Args:
        X (Tensor): Data tensor of shape (N, D)
        w (Tensor): weight tensor of shape (D, )
        b (Tensor): bias tensor of shape (, )

    Returns:
        predictions (Tensor): The linear predictions of shape (N, )
    """
    ################################################################
    # TODO
    predictions = X @ w + b
    ################################################################

    return predictions

def calculate_distance_vec(X: torch.Tensor):

    """
    Calculates the pairwise distance matrix for the given input data.

    Args:
        X (Tensor): Data tensor of shape (N, D)

    Returns:
        dist (Tensor): The pairwise distance matrix of shape (N, N)
    """
    
    ################################################################
    # TODO
    dist = torch.sqrt(torch.sum((X.unsqueeze(0) - X.unsqueeze(1))**2, dim = -1))
    ################################################################

    return dist


