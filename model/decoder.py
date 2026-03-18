import math
import torch
import torch.nn as nn
from modules.positional_encoding import sinusoidal_positional_encoding
from blocks.decoder_layer import TransformerDecoderBlock


class TransformerDecoder(nn.Module):
    def __init__(self,  
                 vocab_size, 
                 d_model, 
                 num_layers, 
                 num_heads, 
                 hidden, 
                 max_seq_len, 
                 drop_prob=0.1):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)
        
        # 堆叠Decoder块
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, hidden, drop_prob) 
            for _ in range(num_layers)
        ])

        # 输出层
        self.norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size, bias=False)

        # 权重绑定：输入嵌入和输出层共享权重（减少参数，提升性能）
        self.token_embed.weight = self.output_layer.weight
        # Dropout增强
        self.dropout = nn.Dropout(drop_prob)
        # 初始化
        self._init_weights()

    def _init_weights(self):
        """改进的权重初始化"""
        # 嵌入层初始化
        nn.init.normal_(self.token_embed.weight, std=0.02)
        # 各层初始化
        for layer in self.layers:
            # 注意力投影层
            nn.init.xavier_uniform_(layer.attn.proj_q.weight)
            nn.init.xavier_uniform_(layer.attn.proj_k.weight)
            nn.init.xavier_uniform_(layer.attn.proj_v.weight)
            nn.init.xavier_uniform_(layer.attn.proj_out.weight)
            # 前馈网络层
            nn.init.kaiming_normal_(layer.ffn.linear1.weight)
            nn.init.kaiming_normal_(layer.ffn.linear2.weight)

    def create_causal_mask(self, seq_len):
        """
        创建因果掩码（下三角矩阵）
        """
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        return mask  # [seq_len, seq_len]

    def forward(self, input_ids):
        """
        参数:
        - input_ids: 输入的token IDs [batch_size, seq_len]
        
        返回:
        - logits: 预测的logits [batch_size, seq_len, vocab_size]
        - all_attn_weights: 各层的注意力权重
        """
        batch_size, seq_len = input_ids.shape
        # 嵌入层
        token_embeds = self.token_embed(input_ids)  # [batch, seq, d_model]
        token_embeds = token_embeds * math.sqrt(self.d_model)  # 缩放嵌入
        # 位置编码
        embeddings = sinusoidal_positional_encoding(token_embeds)
        embeddings = self.dropout(embeddings)
        # 创建因果掩码
        causal_mask = self.create_causal_mask(seq_len).to(input_ids.device)
        causal_mask = causal_mask.unsqueeze(0)  # [1, seq_len, seq_len]
        # 通过所有Decoder层
        hidden_states = embeddings
        all_attn_weights = []
        for layer in self.layers:
            hidden_states, attn_weights = layer(hidden_states, causal_mask)
            all_attn_weights.append(attn_weights.detach())  # 保存权重用于可视化

        # 最终归一化
        hidden_states = self.norm(hidden_states)
        # 输出层
        logits = self.output_layer(hidden_states)

        return logits, all_attn_weights
    

if __name__ == "__main__":
    model = TransformerDecoder(
        vocab_size=500,
        d_model=256,
        num_layers=12,
        num_heads=8,
        hidden=256*4,
        max_seq_len=128,
        drop_prob=0.1
    ) 

    batch_size, seq_len = 16, 10
    x = torch.randint(0, 500, (batch_size, seq_len))

    print(f"input size is {x.size()}")
    print(f"score is {x[0]}")

    output, attn_w = model(x)

    print(f"output size is {output.size()}")
    print(f"output is {output[0]}")


