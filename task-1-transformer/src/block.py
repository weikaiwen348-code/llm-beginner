

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
    def __init__(self, d_model, num_heads ,d_hidden):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_hidden = d_hidden
        
        
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.FFN = nn.Sequential(nn.Linear(d_model, d_hidden),
                                 nn.GELU(),
                                 nn.Linear(d_hidden, d_model)
        )
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        

    def forward(self, x, mask = None) :
        
        x = x + self.attention(x, mask)
        
        x = self.ln1(x)
        
        x = x + self.FFN(x)
        
        x = self.ln2(x)
        
        return x
    
    
if __name__ == "__main__":
    # test
    B = 2
    T = 10
    d_model = 128
    num_heads = 4
    d_hidden = 512
    device = torch.device("mps")
    x = torch.randn(B, T , d_model).to(device)


    Block = TransformerBlock(d_model , num_heads , d_hidden)
    Block.to(device)

    print("input shape:", x.shape)
    print("output shape:", Block(x).shape)
    