import torch
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    """位置式前馈网络（带GeLU激活函数）"""
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.activation = nn.GELU()   # GeLU激活函数
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        # 残差链接
        x = self.dropout(x)
        x = self.linear2(x)        
        return x
    
    
if __name__ == "__main__":
    batch_size, seq_len, hidden_size = 16, 10, 768
    x = torch.randn(batch_size, seq_len, hidden_size)
    print(f"intput size is {x.size()}")
    print(f"score is {x[0]}")
    pffn = PositionwiseFeedForward(hidden_size, hidden_size*4)

    output = pffn(x)

    print(f"output size is {output.size()}")
    print(f"score is {output[0]}")