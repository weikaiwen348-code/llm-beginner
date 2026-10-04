import torch
from torch import nn
from .attention import MultiHeadAttention

class LayerNorm(nn.Module) :
    # x(B ,T , d_model)
    def __init__(self, d_model, eps = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        

    def forward(self, x):
        # 算方差 / 均值
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        y = (x - mean) / ((var + self.eps) ** 0.5)
        return self.gamma * y + self.beta

class TransformerBlock(nn.Module) :
    def __init__(self, d_model, num_heads ,d_hidden , dropout = 0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_hidden = d_hidden
        
        
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.FFN = nn.Sequential(nn.Linear(d_model, d_hidden),
                                 nn.GELU(),
                                 nn.Linear(d_hidden, d_model)
        )
        self.dropout = nn.Dropout(p = dropout)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        

    def forward(self, x, kv_cache=None, return_cache=False) :
        
        attention_input = self.ln1(x)
        
        if return_cache:
            attention_output, new_cache = self.attention(
                attention_input ,
                kv_cache=kv_cache,
                return_cache=True,
            )
        else:
            attention_output = self.attention(
                attention_input ,
                kv_cache=kv_cache,
                return_cache=False,
            )
            
        x = x + self.dropout(attention_output)
        
        ffn_input = self.ln2(x)
                
        x = x + self.dropout(self.FFN(ffn_input))

        if return_cache:
            return x, new_cache
        return x
        
        
        

    
    