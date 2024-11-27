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
            
        """
            d_model : embedding dimensions 
            vocab_size : unique vocab words/tokens 
            x : input matrix (batch_size = 1, seq_len)
            output matrix : (1,seq_len, d_model)
    
        """
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


        
        """
        x: input matrix of shape (1,6, 512)
        return matrix : positional embedded matrix of shape (1,6,512)

        Notes: 
        x shape would be (batch_size, seq_length, embedding_dim)
        We are slicing the pe since the input length could be of varying size 
        Padding is alternative instead of slicing. But that would unnecessary computation and wasting the memory
        Better to use slicing especially when the seq_len = 512, the padding would cause significant computation and memory wastage 
        """
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)
        

class LayerNormalization (nn.Module):

    def __init__(self, eps:float = 10 ** -6) -> None : 
        
        super().__init__()
        self.eps = eps 
        self. alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self. bias = nn.Parameter(torch.zeros(1)) # Added 

    def forward (self, x):
        """
        x : input matrix (1,6,512)
        output matrix : (1,6,512)
        """

        mean = x.mean(dim = -1, keepdim = True )
        std = x.std(dim = -1, keepdim = True )

        return self.alpha * (x - mean) / (std + self.eps ) + self.bias 
    

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model:int, d_ff:int, dropout:float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self,x):
        
        #(Batch = 1, seq_len, d_model) -> (1, seq_len, d_ff) --> (1,seq_len,d_model)
        return self.linear_2(self.dropout(nn.ReLU(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model:int, h:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h 
        self.dropout = nn.Dropout(dropout)

        assert d_model % h == 0 , "d_model isn't divisble by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) #Wq 
        self.w_k = nn.Linear(d_model,d_model) #Wk
        self.w_v = nn.Linear(d_model, d_model) #Wv

        self.w_o = nn.Linear(d_model,d_model) #Wo

    @staticmethod
    def attention (query, key, value , mask, dropout:nn.Dropout):
        d_k = query.shape[-1]

        # (Batch, h, seq_len, d_k) -> (Batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)

        if mask is not None: 
            attention_scores.masked_fill(mask==0, -1e9)
        
        attention_scores = attention_scores.softmax(dim = -1) # (Batch, h, seq_len, seq_len)

        if dropout is not None: 
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value) , attention_scores 
    
    def forward( self, q, k, v , mask):

        query = self.w_q(q) #(Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        key = self.w_k(k) #(Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        value = self.w_v(v) #(Batch, seq_len, d_model) --> (Batch, seq_len, d_model)

        #(Batch, seq_len, d_model) --> (Batch, seq_len, h, d_k) --transpose--> (Batch,h, seq_len, d_k)
        #We are trying to split the embedding and learn the attention in the small embedding in parallel and aggregate the learning at the end 
        query = query.view(query.shape[0],query.shape[1],self.h, self.d_k ).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.h, self.d_k ).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.h, self.d_k ).transpose(1,2)

        x, attention_scores = MultiHeadAttentionBlock.attention(query,key,value,mask)

        # (Batch, h, seq_len, d_k ) --> (Batch, seq_len, h, d_k) --> (Batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1, self.h * self.d_k)

        #(Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        return self.w_o(x)
    

class ResidualConnection(nn.module):

    def __init__(self, dropout:float) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))







    


    
