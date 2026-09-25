def create_patch_sequence(x, patch_size=16):

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

    batch_size, channels, height, width = x.shape
    assert height % patch_size == 0 and width % patch_size == 0, "Image dimensions must be divisible by the patch size."
    
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    num_patches = num_patches_h * num_patches_w
    patch_dim = channels * patch_size * patch_size

    # loop based implementation (easier to understand)

    # sequence = torch.zeros(batch_size, num_patches, patch_dim, device=x.device)
    # for i in range(0, height, patch_size):
    #    for j in range(0, width, patch_size):
    #        patch = x[:, :, i:i+patch_size, j:j+patch_size].reshape(-1, 3 * patch_size * patch_size)
    #        sequence[:, (i//patch_size) * (width // patch_size) + (j//patch_size), :] = patch
    
    # vectorized implementation using unfold (more efficient)
    sequence = (
        x.unfold(2, patch_size, patch_size)
        .unfold(3, patch_size, patch_size)
        .permute(0, 2, 3, 1, 4, 5)
        .reshape(x.size(0), -1, x.size(1) * patch_size * patch_size)
    )

    return sequence