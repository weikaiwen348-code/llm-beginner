# 检验：python eval/run.py

import torch

import torch.nn as nn

def scaled_dot_product_attention(Q, K, V, mask=None, return_weights=False) :
    #Q(B , H , T , d_k) , K(B , H , T , d_k) , V(B , H , T , d_v)
    
    # scores(B , H , T , T) = Q(B , H , T , d_k) @ K(B , H , d_k , T)
    scores = Q @ K.transpose(-2 , -1)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 1, float('-inf'))
        
    scale = Q.shape[-1] ** 0.5
    scores = scores / scale
    attention_weights = torch.softmax(scores , dim = -1)
    
    output = attention_weights @ V
    
    if return_weights:
        return output, attention_weights
    return output

class MultiHeadAttention(nn.Module) :
    
    def __init__(self , d_model , num_heads) :
        super(MultiHeadAttention , self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0
        
        self.depth = d_model // num_heads
        
        self.rope = RotaryEmbedding(self.depth)
        
        self.Wq = nn.Linear(d_model , d_model)
        self.Wk = nn.Linear(d_model , d_model)
        self.Wv = nn.Linear(d_model , d_model)
        
        self.W_out = nn.Linear(d_model , d_model)
        
    def split_heads(self , x) :
        # x(B , T , d_model) -> (B , H , T , d_k)
        return x.view(x.shape[0] , -1 , self.num_heads , self.depth).transpose(1 , 2)
                
        
    def forward(self , x , mask=None, return_weights=False) :
        # x(B , T , d_model)
        batch_size = x.shape[0]
        Q = self.Wq(x)  # (B , T , d_model)
        K = self.Wk(x)  # (B , T , d_model)
        V = self.Wv(x)  # (B , T , d_model)
        
        Q = self.split_heads(Q)  # (B , H , T , d_k)
        K = self.split_heads(K)  # (B , H , T , d_k)
        V = self.split_heads(V)  # (B , H , T , d_v)
        
        Q = self.rope(Q)
        K = self.rope(K)
        
        output, attention_weights = scaled_dot_product_attention(
            Q, K, V, mask, return_weights=True
        )
        
        output = output.transpose(1 , 2).contiguous().view(batch_size , -1 , self.d_model)
        output = self.W_out(output)
        
        if return_weights:
            return output, attention_weights
        return output



class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()

        assert dim % 2 == 0

        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, dim, 2).float() / dim
            )
        )

        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False
        )

    def forward(self, x):
        # x: (B, H, T, D)

        T = x.shape[-2]

        positions = torch.arange(
            T,
            device=x.device,
            dtype=self.inv_freq.dtype
        )

        # (T, D/2)
        angles = torch.outer(
            positions,
            self.inv_freq
        )

        cos = angles.cos().to(dtype=x.dtype)
        sin = angles.sin().to(dtype=x.dtype)

        # (1, 1, T, D/2)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # (B, H, T, D)
        # ->
        # (B, H, T, D/2, 2)
        x = x.reshape(
            *x.shape[:-1],
            x.shape[-1] // 2,
            2
        )

        x1 = x[..., 0]
        x2 = x[..., 1]

        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        x = torch.stack(
            [rotated_x1, rotated_x2],
            dim=-1
        )

        # 恢复 (B,H,T,D)
        x = x.flatten(-2)

        return x