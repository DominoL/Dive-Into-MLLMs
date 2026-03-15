import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention
    计算注意力权重的公式
    attention(Q,K,V) = softmax((Q*K^T)/sqrt(d_model))*V
    """
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None):
        """
        进行注意力权重的计算
        :param query: 查询向量，[batch_size, seq_len, d_model]
        :param key: 键值向量，[batch_size, seq_len. d_model]
        :param value: 值向量，[batch_size, seq_len, d_model]
        :param mask: 上三角掩码，从普通自注意力变为因果自注意力
        :return: 
            attention_weight:注意力权重 [batch_size, seq_len, seq_len]
            output 注意力输出 [batch_size, seq_len, d_model]
        """
        d_k = key.size()[-1]
        scaled = torch.tensor(d_k, dtype=torch.float32, device=query.device)
        # [batch_size, seq_len, seq_len]
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(scaled)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)
        
        scores = torch.softmax(scores, dim=-1)
        # [batch_size, seq_len, d_model]
        output = torch.matmul(scores, value)
        return output, scores
    
class MultiHeadAttention(nn.Module):
    '''
    multi head attention 模块
    将输入分割为多个头（heads），在每个头上独立计算Scaled Dot-Product Attention，
    最后将结果拼接并通过线性变换输出
    '''
    def __init__(self, d_model, num_heads):
        '''
        :param d_model: 模型维度 
        :param num_heads: 注意力的头数
        '''
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model%num_heads==0, "d_model must be divisiable by num_heads"
        
        self.head_dim = self.d_model // self.num_heads
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)
        self.proj_out = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()

    def split_heads(self, x: torch.Tensor):
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
    def combine_heads(self, x: torch.Tensor):
        '''
        :param x: [batch_size, num_heads, seq_len, head_dim]
        :return: 
        '''
        batch_size, num_heads, seq_len, head_dim = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads*head_dim)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        batch_size, seq_len, d_model = x.size()
        query = self.proj_q(x)
        key = self.proj_k(x)
        value = self.proj_v(x)
        query_splited = self.split_heads(query)
        key_splited = self.split_heads(key)
        value_splited = self.split_heads(value)
        if mask is not None:
            mask = mask.unsqueeze(1)
        attention_out, attention_weights = self.attention(query_splited, key_splited, value_splited, mask)
        # attention_out [batch_size, num_heads, seq_len, head_dim]
        attention_out = self.combine_heads(attention_out)
        return self.proj_out(attention_out), attention_weights
    
class MaskedMultiHeadAttention(MultiHeadAttention):
    """Masked Multi-Head Attention模块
    专用于Transformer decoder的自注意力层，
    添加未来掩码（future-masking）以防止位置i关注位置j>i的token。
    """
    def __init__(self, d_model, num_heads):
        super(MaskedMultiHeadAttention, self).__init__(d_model, num_heads)

    def forward(self, x: torch.Tensor):
        # 序列长度
        seq_len = x.size(1)
        # 创建未来掩码（下三角矩阵，对角线为1，上三角为0）
        # [1, seq_len, seq_len]
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
        # 调用父类的forward方法并传入掩码
        return super().forward(x, mask)
    

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads=256, drop_prob=0.1):
        super(MultiQueryAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0, "d_model must divisible by num_heads"
        self.head_dim = d_model // num_heads

        # 定义线性变换层
        self.proj_q = nn.Linear(d_model, self.d_model)
        self.proj_k = nn.Linear(d_model, self.head_dim)
        self.proj_v = nn.Linear(d_model, self.head_dim)
        # 定义输出变换层
        self.proj_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(drop_prob)

        # # 初始化位置编码
        # self.positional_encoding = nn.Parameter(torch.randn(num_heads, self.head_dim))
        # 设置缩放因子
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor=None):
        batch_size, seq_len, d_model = x.size()
        # 计算键和值
        key = self.proj_k(x).unsqueeze(1)  # batch_size, 1,seq_len, head_dim
        val = self.proj_v(x).unsqueeze(1)  # batch_size, 1, seq_len, head_dim
        # 计算共享查询
        query = self.proj_q(x)
        # [batch_size, num_heads, seq_length, head_dim]
        Q = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # attention
        scores = torch.matmul(Q, key.transpose(-2, -1)) * self.scale

        # 应用注意力掩码
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)

        # 计算注意力权重
        scores = torch.softmax(scores, dim=-1)  # batch_size, seq_len, seq_len
        # 应用dropout
        scores = self.dropout(scores)
        # 加权求和
        output = torch.matmul(scores, val)   # batch_size, num_heads, seq_len, head_dim
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        # 对输出进行线性变换
        output = self.proj_out(output)

        return output, scores


class GroupQueryAttention(nn.Module):
    def __init__(self, d_model, num_q_heads, num_kv_groups=None, drop_prob=0.1):
        super(GroupQueryAttention, self).__init__()
        self.d_model = d_model
        self.num_kv_groups = num_kv_groups
        self.num_q_heads = num_q_heads
        
        assert d_model % num_q_heads == 0, "d_model must divisible by num_q_heads"
        assert num_q_heads % num_kv_groups == 0, "num_q_heads must be divisible by num_kv_groups"

        self.head_dim = d_model // num_q_heads
        self.proj_q = nn.Linear(d_model, num_q_heads * self.head_dim)
        self.proj_k = nn.Linear(d_model, num_kv_groups * self.head_dim)
        self.proj_v = nn.Linear(d_model, num_kv_groups * self.head_dim)
        self.proj_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(drop_prob)

        # # 初始化位置编码
        # self.positional_encoding = nn.Parameter(torch.randn(num_heads, self.head_dim))
        # 设置缩放因子
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor=None):
        batch_size, seq_len, _ = x.size()
        Q = self.proj_q(x)  # batch_szie, seq_len, d_model
        K = self.proj_k(x)  # batch_size, seq_len, num_kv_groups*head_dim
        V = self.proj_v(x)  # batch_size, seq_len, num_kv_groups*head_dim

        # Q*K.T
        Q = Q.view(batch_size, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)
        K = torch.repeat_interleave(K, self.num_q_heads // self.num_kv_groups, 1) # batch_size, num_q_heads, seq_len, head_dim

        # batch_size, num_q_heads, seq_len, seq_len
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # 应用注意力掩码
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)

        scores = torch.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        
        V = V.view(batch_size, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)
        # batch_size, num_q_heads, seq_len, head_dim
        V = torch.repeat_interleave(V, self.num_q_heads // self.num_kv_groups, 1)
        
        attn_out = torch.matmul(scores, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_q_heads*self.head_dim)

        output = self.proj_out(attn_out)

        return output, scores


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model=1024, n_heads=8, d_k=128, d_c=32, d_hR=32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_c = d_c
        self.d_hR = d_hR

        # 权重矩阵
        self.W_DQ = nn.Linear(d_model, d_c)
        self.W_UQ = nn.Linear(d_c, n_heads * d_k)
        self.W_DKV = nn.Linear(d_model, d_c)
        self.W_UK = nn.Linear(d_c, n_heads * d_k)
        self.W_UV = nn.Linear(d_c, n_heads * d_k)
        self.W_QR = nn.Linear(d_c, n_heads * d_hR)
        self.W_KR = nn.Linear(d_model, d_hR)

        self.W_O = nn.Linear(n_heads * d_k, d_model)

    def apply_rope(self, x, seq_len):
        return x

    def forward(self, h_t, mask=None):
        batch_size, seq_len, _ = h_t.shape

        c_t_Q = self.W_DQ(h_t)
        q_t_C = self.W_UQ(c_t_Q).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1,2)

        c_t_KV = self.W_DKV(h_t)
        k_t_C = self.W_UK(c_t_KV).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1,2)
        v_t_C = self.W_UV(c_t_KV).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1,2)

        q_t_R = self.W_QR(c_t_Q)
        q_t_R = self.apply_rope(q_t_R)

        k_t_R = self.W_KR(h_t).view(batch_size, seq_len, 1, self.d_k).transpose(1,2)
        k_t_R = self.apply_rope(k_t_R)
        k_t_R = k_t_R.expand(-1, self.n_heads, -1, -1)

        q_t = torch.concat([q_t_C, q_t_R], dim=-1)
        k_t = torch.concat([k_t_C, k_t_R], dim=-1)

        attn_scores = torch.matmul(q_t, k_t.transpose(-1, -2)) / math.sqrt(self.dk+self.d_hR)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        o_t = torch.matmul(attn_weights, v_t_C)  # b, s, nh, dk
        o_t = o_t.transpose(1,2).contiguous().view(batch_size, seq_len, self.n_heads*self.d_k)
        u_t = self.W_O(o_t)

        return u_t

if __name__ == "__main__":
    batch_size, seq_len, d_model = 16, 20, 768
    # attention = ScaledDotProductAttention()
    
    # query = torch.randn(batch_size, seq_len, d_model)
    # key = torch.randn(batch_size, seq_len, d_model)
    # value = torch.randn(batch_size, seq_len, d_model)
    # # output, scores = attention(query, key, value)

    # mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
    # print(mask)
    # print(mask.size())
    # output, scores = attention(query, key, value, mask=mask)

    # print(f"output size is {output.size()}")
    # print(f"score size is {scores.size()}")
    # print(f"score is {scores}")
    # print(f"{sum(scores[0][1])}")

    x = torch.randn(batch_size, seq_len, d_model)

    # mmha = MaskedMultiHeadAttention(d_model, num_heads=12)
    # output, scores = mmha(x)
    # print(f"output size is {output.size()}")
    # print(f"score size is {scores.size()}")
    # print(f"score is {scores[0]}")

    mqa = MultiQueryAttention(d_model, num_heads=12)
    output, _ = mqa(x)
    print(f"output is {output.size()}")

    # gqa = GroupQueryAttention(d_model, num_q_heads=12, num_kv_groups=4)
    # output, _ = gqa(x)
    # print(f"output is {output.size()}")



