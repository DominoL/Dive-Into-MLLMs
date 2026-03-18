import math
import torch


def sinusoidal_positional_encoding(x):
    batch_size, seq_len, head_dim = x.size()
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, head_dim).float() * (-math.log(1000) / head_dim))
    pe = torch.zeros(seq_len, head_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return x + pe.unsqueeze(0)

def rotate_positional_encoding(x):
    batch_size, seq_len, head_dim = x.size()
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, head_dim).float() * (-math.log(1000) / head_dim))
    cos = torch.cos(position * div_term)
    sin = torch.sin(position * div_term)
    x1, x2 = x[:, 0::2], x[:, 1::2]
    rotated_x1 = x1*cos - x2*sin
    rotated_x2 = x1*sin + x2*cos
    return torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)
