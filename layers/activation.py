import torch

def softmax(x, temparature=1.0, dim=-1):
    # temparature温度越高分布越均匀，温度越低分布更集中，熵更小
    # 数值稳定性处理：减少最大值防止指数爆炸
    max_x = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp((x-max_x) / temparature)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp


if __name__ == "__main__":
    batch_size, seq_len = 4, 10
    x = torch.randn(batch_size, seq_len)
    res = softmax(x, temparature=0.5, dim=-1)
    print(f"result is {res}")
    print(f"sum of result is {torch.sum(res, dim=-1)}")