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

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        # weight sharing trick for query, key and value projections
        # could also be implemented as three separate linear layers 
        self.qkv_proj = nn.Linear(model_dim, 3 * model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, x):

        """
        Forward pass of the Multi Head Self Attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, model_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, model_dim).
            torch.Tensor: Attention weights of shape (batch_size, num_heads, seq_len, seq_len).
        """

        batch_size, seq_len, _ = x.shape

        qkv = self.qkv_proj(x)  # Shape: (batch_size, seq_len, 3 * model_dim)

        Q, K, V = qkv.chunk(3, dim=-1)  # Each of shape: (batch_size, seq_len, model_dim)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # Shape: (batch_size, num_heads, seq_len, head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # Shape: (batch_size, num_heads, seq_len, head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # Shape: (batch_size, num_heads, seq_len, head_dim)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # Shape: (batch_size, num_heads, seq_len, seq_len)
        
        attn_weights = torch.softmax(attn_scores, dim=-1)  # Shape: (batch_size, num_heads, seq_len, seq_len)
        
        attn_output = torch.matmul(attn_weights, V)  # Shape: (batch_size, num_heads, seq_len, head_dim)    

        attn_output = attn_output.transpose(1, 2).contiguous()  # Shape: (batch_size, seq_len, num_heads, head_dim)

        attn_output = attn_output.view(batch_size, seq_len, self.model_dim)  # Shape: (batch_size, seq_len, model_dim)

        output = self.out_proj(attn_output)  # Shape: (batch_size, seq_len, model_dim)  

        return output, attn_weights