import torch
from torch import nn
from .block import TransformerBlock

# 输出二分类
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, num_heads ,d_hidden, num_layers , pad_id=1):
        super().__init__()
        
        self.pad_id = pad_id
        self.max_len = max_len
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_hidden)for _ in range(num_layers)])
        # blocks 出来【B，T ，d_moedl】
        
        self.classifier = nn.Linear(d_model , 2)
        
    def forward(self, input_ids, mask=None):
        
        input_ids = input_ids[:, :self.max_len]

        if mask is not None:
            mask = mask[:, :self.max_len]
        else:
            mask = (input_ids == self.pad_id) # 默认pad_id = 1 ,也就是如果x的那个位置是1 就是1 ，否则为0 ，生成mask
                
        # token_embedding
        x = self.token_embedding(input_ids)
        
        # x: (B, T, d_model)
        B = x.shape[0]
        T = x.shape[1]
        
        # position_embedding
        position_ids = torch.arange(T, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(B, T)      #ai查的
        
        position_vector = self.position_embedding(position_ids)
        
        x = x + position_vector
        
        # padding mask reshape mask[B , T] ---> [B, 1, 1, T]
        attention_mask = mask.unsqueeze(1).unsqueeze(2)
        
        #  遍历blocks
        for block in self.blocks:
            x = block(x, attention_mask)
            
        # mean pooling [B,T,d_model] --> [B , d_model]
        
        valid_mask = (mask == 0)
        valid_mask = valid_mask.unsqueeze(-1)      #[B,T,1]
        x = x * valid_mask 
        
        sum_x = x.sum(dim=1)                       # (B, d_model)
        
        token_count = valid_mask.sum(dim=1)        # (B, 1)
        
        output = sum_x / token_count.clamp(min=1)
        
        # classifier
        output = self.classifier(output)
        
        return output

        
        
        
    
        
        
