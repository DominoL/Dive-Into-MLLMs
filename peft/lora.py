import torch
import torch.nn as nn


class LoraLayer(nn.Module):
    def __init__(self, linear: nn.Linear, rank=32, alpha=32):
        super(LoraLayer, self).__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha

        self.lora_A = nn.Parameter(torch.randn(linear.in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, linear.out_features))

        self.linear.weight.requires_grad = False

    def forward(self, x: torch.Tensor):
        y_linear = self.linear(x)
        y_lora = x.matmul(self.lora_A).matmul(self.lora_B)
        return y_linear + (self.alpha / self.rank) * y_lora
