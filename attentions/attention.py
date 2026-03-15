import math
import torch
import torch.nn as nn


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
        d_k = query.size()[-1]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)

        scores = torch.softmax(scores, dim=-1)
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
        assert d_model % num_heads == 0, "d_model must be divisiable by num_heads"

        self.head_dim = self.d_model // self.num_heads
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)
        self.proj_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()

    def split_heads(self, x: torch.Tensor):
        '''
        :param x: [batch_size, seq_len, d_model]
        :return: [batch_size, num_heads, seq_len, head_dim]
        '''
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor):
        '''
        :param x: [batch_size, num_heads, seq_len, head_dim]
        :return: [batch_size, seq_len, d_model]
        '''
        batch_size, num_heads, seq_len, head_dim = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads*head_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor=None):
        query = self.proj_q(x)
        key = self.proj_k(x)
        value = self.proj_v(x)
        query_splited = self.split_heads(query)
        key_splited = self.split_heads(key)
        value_splited = self.split_heads(value)

        if mask is not None:
            mask = mask.unsqueeze(1)

        out, scores = self.attention(query_splited, key_splited, value_splited, mask=mask)
        out = self.combine_heads(out)
        return self.proj_o(out), scores


class MaskedMultiHeadAttention(MultiHeadAttention):

    def __init__(self, d_model, num_heads):
        super(MaskedMultiHeadAttention, self).__init__(d_model, num_heads)

    def forward(self, x):
        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
        return super().forward(x, mask)



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
    # print(f"score is {scores[0]}")
    # print(f"{sum(scores[0][1])}")

    x = torch.randn(batch_size, seq_len, d_model)

    # mha = MultiHeadAttention(d_model, num_heads=12)
    mmha = MaskedMultiHeadAttention(d_model, num_heads=12)
    output, scores = mmha(x)
    print(f"output size is {output.size()}")
    print(f"score size is {scores.size()}")
    print(f"score is {scores[0]}")


