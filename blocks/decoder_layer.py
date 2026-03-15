import torch
import torch.nn as nn
from layers_old.attention import MultiHeadAttention
from layers_old.feed_forward import PositionwiseFeedForward


class TransformerDecoderBlock(nn.Module):
    """
    纯Decoder结构的Transformer块
    """
    def __init__(self, d_model, num_heads, hidden, drop_prob=0.1):
        super(TransformerDecoderBlock, self).__init__()
        # 核心模块
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, hidden)
        # 前馈网络后的额外归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Dropout增强
        self.dropout1 = nn.Dropout(drop_prob)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x, attn_mask=None):
        # 1. compute self attention
        # 自注意力层（带因果掩码）
        z, attn_weights = self.attn(x, attn_mask)
        # 2. add and norm
        z = self.dropout1(z)
        z = self.norm1(x + z)
        _z = z
        # 3. positionwise feed forward network
        # 前馈网络
        z = self.ffn(z)
        # 4. add and norm
        z = self.dropout2(z)
        z = self.norm2(z + _z)
        return z, attn_weights
    

if __name__ == "__main__":
    batch_size, seq_len, hidden_size = 16, 10, 768
    x = torch.randn(batch_size, seq_len, hidden_size)

    print(f"input size is {x.size()}")
    print(f"score is {x[0]}")

    decoder_block = TransformerDecoderBlock(hidden_size, 12, hidden_size*4)
    output, attn_scores = decoder_block(x)
    print(f"output size is {output.size()}")
    print(f"score is {output[0]}")

