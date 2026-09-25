import torch

def create_patch_sequence(x, patch_size=4):

    """
    Converts an image into a sequence of patches.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        patch_size (int): Size of each patch (patch_size x patch_size).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, num_patches, patch_dim),
                      where num_patches = (height // patch_size) * (width // patch_size)
                      and patch_dim = channels * patch_size * patch_size.
    """

    sequence = None

    ################################################################
    # TODO
    batch_size, channels, height, width = x.shape
    assert height % patch_size == 0 or width % patch_size == 0, "error"

    sequence = (
        x.unfold(2, patch_size, patch_size)
        .unfold(3, patch_size, patch_size) 
        .permute(0, 2, 3, 1, 4, 5)
        .reshape(batch_size, -1, channels * patch_size * patch_size)
    )
    ################################################################

    return sequence