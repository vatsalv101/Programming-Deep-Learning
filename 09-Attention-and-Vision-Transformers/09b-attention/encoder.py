from torch import nn
from attention import MultiHeadSelfAttention

class EncoderBlock(nn.Module):

    def __init__(self, model_dim, num_heads, mlp_hidden_dim):

        """
        Creates a Transformer Encoder Block using Multi Head Self Attention.

        Args:
            model_dim (int): Dimension of the input and output features.
            num_heads (int): Number of attention heads.
            mlp_hidden_dim (int): Dimension of the hidden layers in the MLP.
        """

        super().__init__()

        ################################################################
        # TODO
        
        self.ln1 = nn.LayerNorm(model_dim)
        self.mha = MultiHeadSelfAttention(model_dim, num_heads)
        self.ln2 = nn.LayerNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.GELU(),
            nn.Linear(4 * model_dim, model_dim),
        )
        
        ################################################################

    def forward(self, x):

        """
        Forward pass of the Encoder Block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, model_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, model_dim).
        """

        ################################################################
        # TODO
        
        q, v = self.mha(self.ln1(x))
        q = x + q

        z = self.mlp(self.ln2(q))
        z = q + z
        
        return z

        ################################################################


def create_transformer_encoder(num_layers, model_dim, num_heads, mlp_hidden_dim):

    """
    Creates a stack of Transformer encoder blocks.

    Args:
        num_layers (int): Number of Transformer encoder layers.
        model_dim (int): Dimension of the input and output features.
        num_heads (int): Number of attention heads.
        mlp_hidden_dim (int): Dimension of the hidden layers in the MLP.

    Returns:
        nn.Sequential: A sequential container of Transformer encoder blocks.
    """

    encoder = None
    ################################################################
    # TODO

    blocks = [EncoderBlock(model_dim, num_heads, mlp_hidden_dim) for _ in range(num_layers)]
    encoder = nn.Sequential(*blocks)
    ################################################################
    return encoder
