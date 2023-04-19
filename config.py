# define a config class for paramerters
class config:
     
    max_seq_len = 200 # maximum sequence length
    d_model = 512 # model dimension
    n_head = 8 # number of heads
    batch_size = 30 # batch size for input sequence
    n_layer = 6 # number of encoder and decoder layers
    
cfg = config()