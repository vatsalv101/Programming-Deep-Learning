from torch import nn
import torch

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, model_dim, num_heads):

        """
        Creates a Multi Head Self Attention module.

        Args:
            model_dim (int): Dimension of the input and output features.
            num_heads (int): Number of attention heads.
        """

        super().__init__()

        ################################################################
        # TODO
        self.model_dim = model_dim
        self.num_heads = num_heads

        self.head_dim = model_dim // num_heads

        self.qkv = nn.Linear(model_dim, 3 * model_dim)
        self.out = nn.Linear(model_dim, model_dim)
        ################################################################

    def forward(self, x):

        """
        Forward pass of the Multi Head Self Attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, model_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, model_dim).
            torch.Tensor: Attention weights of shape (batch_size, num_heads, seq_len, seq_len).
        """

        ################################################################
        # TODO
        batch_size, seq_len, model_dim = x.shape

        qkv = self.qkv(x)
        Q, K, V = qkv.chunk(3, dim=-1)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_score = torch.matmul(Q, K.transpose(-2, -1)) / self.model_dim ** .5

        attn_weight = torch.softmax(attn_score, dim=-1)

        attn_out = torch.matmul(attn_weight, V).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.out(attn_out), attn_weight
        
        
        ################################################################