# 检验：python eval/run.py

import torch

import torch.nn as nn

def scaled_dot_product_attention(Q, K, V, mask=None) :
    #Q(B , H , T , d_k) , K(B , H , T , d_k) , V(B , H , T , d_v)
    
    # scores(B , H , T , T) = Q(B , H , T , d_k) @ K(B , H , d_k , T)
    scores = Q @ K.transpose(-2 , -1)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 1, float('-inf'))
        
    scale = Q.shape[-1] ** 0.5
    scores = scores / scale
    attention_weights = torch.softmax(scores , dim = -1)
    
    output = attention_weights @ V
    
    return output

class MultiHeadAttention(nn.Module) :
    
    def __init__(self , d_model , num_heads) :
        super(MultiHeadAttention , self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0
        
        self.depth = d_model // num_heads
        
        self.Wq = nn.Linear(d_model , d_model)
        self.Wk = nn.Linear(d_model , d_model)
        self.Wv = nn.Linear(d_model , d_model)
        
        self.W_out = nn.Linear(d_model , d_model)
        
    def split_heads(self , x) :
        # x(B , T , d_model) -> (B , H , T , d_k)
        return x.view(x.shape[0] , -1 , self.num_heads , self.depth).transpose(1 , 2)
                
        
    def forward(self , x , mask=None) :
        # x(B , T , d_model)
        batch_size = x.shape[0]
        Q = self.Wq(x)  # (B , T , d_model)
        K = self.Wk(x)  # (B , T , d_model)
        V = self.Wv(x)  # (B , T , d_model)
        
        Q = self.split_heads(Q)  # (B , H , T , d_k)
        K = self.split_heads(K)  # (B , H , T , d_k)
        V = self.split_heads(V)  # (B , H , T , d_v)
        
        output  = scaled_dot_product_attention(Q , K , V , mask)
        
        output = output.transpose(1 , 2).contiguous().view(batch_size , -1 , self.d_model)
        output = self.W_out(output)
        
        return output
    
  

        
        