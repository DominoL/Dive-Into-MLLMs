import math
import torch
import torch.nn as nn

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads=256, drop_prob=0.1):
        super(MultiQueryAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must divisible by num_heads"
        self.head_dim = d_model // num_heads

        self.proj_q = nn.Linear(d_model, self.d_model)
        self.proj_k = nn.Linear(d_model, self.head_dim)
        self.proj_v = nn.Linear(d_model, self.head_dim)
        self.proj_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(drop_prob)

        self.scale = 1 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, masked=False):
        batch_size, seq_len, _ = x.size()

        k = self.proj_k(x).unsqueeze(1)
        v = self.proj_v(x).unsqueeze(1)
        q = self.proj_q(x)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if masked:
            mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(1)
            scores = mask.masked_fill(mask==0, -1e9)

        scores = torch.softmax(scores, dim=-1)

        scores = self.dropout(scores)
        out = torch.matmul(scores, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        out = self.proj_o(out)
        return out, scores
    

class GroupQueryAttention(nn.Module):
    def __init__(self, d_model, num_q_heads, num_kv_groups, drop_prob=0.1):
        super(GroupQueryAttention, self).__init__()
        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_groups = num_kv_groups

        assert d_model % num_q_heads == 0, "d_model must divisible by num_q_heads"
        assert num_q_heads % num_kv_groups == 0, "num_q_heads must be divisible by num_kv_groups"

        self.head_dim = d_model // num_q_heads
        self.proj_q = nn.Linear(d_model, num_q_heads * self.head_dim)
        self.proj_k = nn.Linear(d_model, num_kv_groups * self.head_dim)
        self.proj_v = nn.Linear(d_model, num_kv_groups * self.head_dim)
        self.proj_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(drop_prob)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, masked=False):
        batch_size, seq_len, _ = x.size()
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)

        q = q.view(batch_size, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)
        k = torch.repeat_interleave(k, self.num_q_heads // self.num_kv_groups, 1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if masked:
            mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(1)
            scores = scores.masked_fill(mask==0, -1e9)
        
        scores = torch.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        v = v.view(batch_size, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)
        v = torch.repeat_interleave(v, self.num_q_heads // self.num_kv_groups, 1)
        out = torch.matmul(scores, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.proj_o(out)

        return out, scores

    

if __name__ == "__main__":
    batch_size, seq_len, d_model = 16, 20, 768
    x = torch.randn(batch_size, seq_len, d_model)

    # mqa = MultiQueryAttention(d_model, num_heads=12)
    # out, scores = mqa(x)

    gqa = GroupQueryAttention(d_model, num_q_heads=12, num_kv_groups=4)
    out, scores = gqa(x, masked=True)
    print(f"output size is {out.size()}")
    print(f"score size is {scores.size()}")
    print(f"score is {scores[0]}")
    print(f"{sum(scores[0][1])}")

