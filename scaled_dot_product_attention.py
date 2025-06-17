from torch import nn
import torch

class ScaledDotProductAttention(nn.Module):
    """Defines scaled dot product attention layer.

      Attributes:
        dropout: Dropout rate to use
        activation: Normalisation function for scaled dot product attention (e.g.
          softmax by default)
    """

    def __init__(self, attn_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()

        self.dropout = nn.Dropout(attn_dropout)
        self.activation = nn.Softmax(dim=-1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, q, k, v, mask):
        """Applies scaled dot product attention.

        Args:
          q: Queries
          k: Keys
          v: Values
          mask: Masking if required -- sets softmax to very large value

        Returns:
          Tuple of (layer outputs, attention weights)
        """

        assert q.dim() == 3, f"Expected 3D tensor, got {q.dim()}"
        assert k.dim() == 3, f"Expected 3D tensor, got {k.dim()}"
        assert v.dim() == 3, f"Expected 3D tensor, got {v.dim()}"
        attn = torch.bmm(q,k.permute(0,2,1)) # shape=(batch, q, k)
        if mask is not None:
            attn = attn.masked_fill(mask.bool().to("cuda"), -1e9)

        attn = self.activation(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn,v)
        return output, attn

# batch_size = 2
# seq_len = 4
# d_k = 8  # 维度
#
# q = torch.rand(batch_size, seq_len, d_k)
# k = torch.rand(batch_size, seq_len, d_k)
# v = torch.rand(batch_size, seq_len, d_k)
# mask = torch.randint(0, 2, (batch_size, seq_len, seq_len))
#
# attn_layer = ScaledDotProductAttention(attn_dropout=0.1)
# output, attn = attn_layer(q, k, v, mask)
#
# print("Output shape:", output.shape)  # 预期: (batch_size, seq_len, d_k)
# print("Attention shape:", attn.shape)  # 预期: (batch_size, seq_len, seq_len)
