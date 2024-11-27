import torch
import torch.nn as nn
import math 

class InputEmbeddings(nn.Module):

    def __init__(self, d_model:int, vocab_size:int ):
        
        super().__init__()
        self.d_model = d_model 
        self.vocab_size = vocab_size 
        self.embedding = nn.Embedding(vocab_size, d_model)

    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) # multiply sqrt of d_model factor as mentioned in the paper

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model:int, seq_len:int, dropout:float) -> None:
        
        super().__init__()
        self.d_model = d_model
        self.seq.len = seq_len 
        self.dropout = nn.Dropout(dropout)

        #Create the matrix of (seq_len,d_model)
        pe = torch.zeros(seq_len,d_model)

        #Create a vector of text length (seq_len, 1)
        position = torch.arange(0,seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (- math.log(10000.0) / d_model))
        
        #Apply sin to odd and cos to even 
        pe[:,0::2] = torch.sin(position *  div_term)
        pe[:,1::2] = torch.cos(position *  div_term)

        #(1, seq_Len, d_model)
        pe = pe.unsequeeze(0)

        self.register_buffer('pe', pe) 

    def forward(self, x):

        # x shape would be (batch_size, seq_length, embedding_dim)
        # We are slicing the pe since the input length could be of varying size 
        # Padding is alternative instead of slicing. But that would unnecessary computation and wasting the memory
        # Better to use slicing especially when the seq_len = 512, the computation and memory wastage would be significant 
        
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)


    

    
