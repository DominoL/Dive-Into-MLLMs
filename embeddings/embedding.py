import torch
import torch.nn as nn
from torch.nn import functional as F

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super(RotaryEmbedding, self).__init__()
        self.base = base
        self.dim = dim
        # 构造每对维度的频率
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))  # [dim/2]
        self.inv_freq = inv_freq

    def sinusoidal_position_emb(self, seq_len: int, device):
        """
        m是token所处的位置
        th_i = 1 / base**(2*i/d)
        i = [0, 1, 2, ..., d/2-1]
        [1/base**0, 1/base**(2*1/d), 1/base**(2*2/d), ..., 1/base**(2*d/2/d)]
        :param seq_len: 序列长度
        :return:
        """
        # 生成token序列位置 pos=[0, 1, ..., seq_len-1]
        pos = torch.arange(seq_len, device=device).type_as(self.inv_freq)  # [seq_len]
        # 计算m*th_i
        emb = torch.outer(pos, self.inv_freq.to(device)) # pos*inv_freq [seq_len, dim // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1) # seq_len * dim//2 * 2
        emb = torch.reshape(emb, (seq_len, -1)) # reshape后：偶数sin，奇数cos
        return emb[None, None, :, :]
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        # x: [bs, num_head, seq_len, head_dim] -> [bs, num_head, seq_len, head_dim//2, 2]
        x_rotate = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1)
        x_rotate = x_rotate.reshape(x.shape)  # reshape后正负交替
        return x_rotate
    
    def apply_rotary_emb(self, x: torch.Tensor, seq_len):
        # x: [bs, num_head, seq_len, head_dim]
        # cos: [seq_len, head_dim]
        pos_emb = self.sinusoidal_position_emb(seq_len=seq_len, device=x.device)
        # 将奇数列cos信息抽取出来
        cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
        # 将偶数列sin信息抽取出来
        sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)
        return x * cos_pos + self.rotate_half(x) * sin_pos
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        return self.apply_rotary_emb(x, seq_len)