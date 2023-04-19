import torch.nn as nn
from multi_head_attention import MultiheadAttention
from config import cfg


class PositionwiseFeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = cfg.d_model
        self.fc1 = nn.Linear(self.d_model, self.d_model * 4)
        self.fc2 = nn.Linear(self.d_model * 4, self.d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # define the parameters
        self.max_seq_len = cfg.max_seq_len
        self.d_model = cfg.d_model
        self.n_head = cfg.n_head
        self.batch_size = cfg.batch_size
        
        # define the positional encoding
        self.MultiHeadAttention = MultiheadAttention(self.d_model, self.n_head)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.feed_forward = PositionwiseFeedForward()
        
    def forward(self, x):
        post_attention = self.MultiHeadAttention(x)
        res_attention = x + post_attention # residual connection of multi-head attention
        post_res_attention = self.layer_norm(res_attention)
        
        post_fc = self.fc(post_res_attention)
        res_fc = post_res_attention + post_fc # residual connection of feed forward
        output = self.layer_norm(res_fc)
        
        return output

class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_layer = cfg.nlayer
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(self.n_layer)])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Alternative implementation using nn.Sequential
# class TransformerEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.n_layer = cfg.n_layer
#         self.layers = nn.Sequential(*[EncoderLayer() for _ in range(self.n_layer)])
         
#     def forward(self, x):
#         x = self.layers(x)
#         return x