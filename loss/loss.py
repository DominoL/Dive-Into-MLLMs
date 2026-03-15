import torch
import torch.nn as nn

def loss_func(x, logits):
    """
    x: [batch_size, seq_len]
    logits: [batch_size, seq_len, val_size]
    """
    shift_labels = x[:, 1:]
    shift_logits = logits[:, :-1, :]

    val_size = shift_logits.size()[-1]

    # [batch_size * (seq_len-1), val_size]
    shift_logits = shift_logits.contiguous().view(-1, val_size)
    # [batch_size * (seq_len-1)]
    shift_labels = shift_labels.contiguous().view(-1)

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(shift_logits, shift_labels)
    return loss


if __name__ == "__main__":
    batch_size, seq_len, val_size = 32, 20, 10000
    x = torch.randint(0, val_size, size=(batch_size, seq_len))
    logits = torch.rand(size=(batch_size, seq_len, val_size))
    logits = torch.softmax(logits, dim=-1)

    l = loss_func(x, logits)
    print(l)

