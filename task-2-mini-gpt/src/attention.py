# 检验：python eval/run.py

import torch

import torch.nn as nn
from .rope import rope
def scaled_dot_product_attention(Q, K, V , mask = None) :
    #Q(B , H , T , d_k) , K(B , H , T , d_k) , V(B , H , T , d_v)
    
    # scores(B , H , T , T) = Q(B , H , T , d_k) @ K(B , H , d_k , T)
    
    scores = Q @ K.transpose(-2 , -1)
    if mask is not None:
        scores = scores.masked_fill(mask , float("-inf"))
    
    scale = Q.shape[-1] ** 0.5
    scores = scores / scale
    attention_weights = torch.softmax(scores , dim = -1)
    
    output = attention_weights @ V
    
    return output

class MultiHeadAttention(nn.Module) :
    
    def __init__(self , d_model , num_heads ) :
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
                
        
    def forward(self , x , kv_cache=None, return_cache=False) :
        if kv_cache is None:
            past_len = 0
        else:
            past_k, past_v = kv_cache
            past_len = past_k.shape[-2]
            
            
        # x(B , T , d_model)
        batch_size = x.shape[0]
        T = x.shape[1]
        
        Q = self.Wq(x)  # (B , T（） , d_model)
        K_new = self.Wk(x)  # (B , T（） , d_model)
        V_new = self.Wv(x)  # (B , T（） , d_model)
        
        Q = self.split_heads(Q)  # (B , H , T , d_k)
        K_new = self.split_heads(K_new)  # (B , H , T , d_k)
        V_new = self.split_heads(V_new)  # (B , H , T , d_v)
        Q, K_new = rope(
                            Q,
                            K_new,
                            position_offset=past_len,
                        )
        
        if kv_cache is None:
            K_total = K_new
            V_total = V_new
        else:
            K_total = torch.cat([past_k, K_new], dim=-2)
            V_total = torch.cat([past_v, V_new], dim=-2)
        T_new = Q.shape[-2]
        T_total = K_total.shape[-2]
            
        # mask矩阵
        query_positions = past_len + torch.arange(
            T_new,
            device=x.device,
        )

        key_positions = torch.arange(
            T_total,
            device=x.device,
        )

        mask = (
            key_positions.unsqueeze(0)
            > query_positions.unsqueeze(1)
        )
        
        output  = scaled_dot_product_attention(Q , K_total , V_total , mask)
        new_cache = (K_total, V_total)
        output = output.transpose(1 , 2).contiguous().view(batch_size , -1 , self.d_model)
        output = self.W_out(output)
        if return_cache:
            return output, new_cache
        return output
    
  

        
        