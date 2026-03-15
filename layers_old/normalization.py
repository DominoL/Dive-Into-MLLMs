import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if type(normalized_shape) == int:
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            gamma = torch.ones(normalized_shape) # 缩放参数
            beta = torch.zeros(normalized_shape) # 偏移参数
            self.gamma = nn.Parameter(gamma)
            self.beta = nn.Parameter(beta)
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def forward(self, x):
        # 计算最后N个维度的均值和方差
        # 例如输入为[B,C,H,W],则对C/H/W计算
        # 例如输入为[B,S,H],则对H计算
        dim = [-i for i in range(1, len(self.normalized_shape)+1)]
        mean = torch.mean(x, keepdim=True, dim=dim)
        # 有偏方差
        var = torch.var(x, keepdim=True, dim=dim, unbiased=False)
        # 归一化
        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        if self.elementwise_affine:
            return self.gamma * x_hat + self.beta
        else:
            return x_hat
        

if __name__ == "__main__":
    batch_size, seq_length, hidden_size = 16, 20, 768
    x = torch.randn(batch_size, seq_length, hidden_size)
    ln = LayerNorm(hidden_size)
    res = ln(x)

    print("输入形状：", x.shape)
    print("输入均值：", x.mean(dim=-1))
    print("输入方差：", x.var(dim=-1, unbiased=False))

    print("输出形状：", res.shape)
    print("输出均值：", res.mean(dim=-1))
    print("输出方差：", res.var(dim=-1, unbiased=False))




