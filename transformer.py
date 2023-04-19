import torch
import torch.nn as nn
from config import cfg
from encoder import TransformerEncoder
from decoder import TransformerDecoder



# construct a transformer model as a class
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # define the parameters
        self.d_model = cfg.d_model
        self.n_head = cfg.n_head
        self.batch_size = cfg.batch_size
        self.nlayer = cfg.nlayer
                
        # define the encoder and decoder
        self.encoder = TransformerEncoder()
        self.decoder = TransformerDecoder()
        self.fc = nn.Linear(self.d_model, )
        
