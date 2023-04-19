import torch
import torch.nn as nn
from config import cfg


class PositionalEncoding(nn.Module):
    def __init__(self):
        """
        positional encoding by sinusoidal function
        
        max_seq_len: maximum sequence length of the input
        d_model: dimension of the model
        encoding: (max_seq_len, d_model)
        """
        super().__init__()
        self.max_seq_len = cfg.max_seq_len
        self.d_model = cfg.d_model
        
        self.encoding = torch.zeros(self.max_seq_len, self.d_model, requires_grad=False)
        postion = torch.arange(0, self.max_seq_len, dtype=float).unsqueeze(1)
        denominator = 10000 ** (torch.arange(0, self.d_model, 2, dtype=float) / self.d_model)
        self.encoding[:, 0::2] = torch.sin(postion / denominator)
        self.encoding[:, 1::2] = torch.cos(postion / denominator)
        
        
    def forward(self, x):
        """
        Input:
        - x (batch_size, seq_len, d_model)
        Output:
        - encoding (seq_len, d_model)
        """
        seq_len = x.size(1)
        
        return self.encoding[:seq_len, :]