import torch
from torch import nn
from .block import TransformerBlock
from .tokenizer import Tokenizer

# 输出二分类
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, num_heads ,d_hidden, num_layers , pad_id=1, dropout =0.0 ):
        super().__init__()
        
        self.pad_id = pad_id
        self.max_len = max_len
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_hidden, dropout=dropout)for _ in range(num_layers)])
        # blocks 出来【B，T ，d_moedl】
        self.embedding_dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Linear(d_model , 2)
        self.final_ln = nn.LayerNorm(d_model)
    def forward(self, input_ids, mask=None, return_attentions=False):
        
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
        
        
        # padding mask reshape mask[B , T] ---> [B, 1, 1, T]
        attention_mask = mask.unsqueeze(1).unsqueeze(2)
        
        #  遍历blocks
        all_attentions = []
        for block in self.blocks:
            if return_attentions:
                x, attention_weights = block(
                    x, attention_mask, return_attention=True
                )
                all_attentions.append(attention_weights)
            else:
                x = block(x, attention_mask)
        x = self.final_ln(x)
        # mean pooling [B,T,d_model] --> [B , d_model]
        
        valid_mask = (mask == 0)
        valid_mask = valid_mask.unsqueeze(-1)      #[B,T,1]
        x = x * valid_mask 
        
        sum_x = x.sum(dim=1)                       # (B, d_model)
        
        token_count = valid_mask.sum(dim=1)        # (B, 1)
        
        output = sum_x / token_count.clamp(min=1)
        
        # classifier
        output = self.classifier(output)
        
        if return_attentions:
            return output, all_attentions
        return output


def load_for_eval(ckpt_path):
    """Restore the model and tokenizer required by ``eval/run.py``."""
    checkpoint = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "checkpoint 使用旧格式，只包含 state_dict；请用新版 train.py 重新训练"
        )

    model_config = checkpoint["model_config"]
    model = TransformerClassifier(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])

    tokenizer_state = checkpoint["tokenizer"]
    tokenizer = Tokenizer(
        pad_id=tokenizer_state["pad_id"],
        unk_id=tokenizer_state["unk_id"],
    )
    tokenizer.string_to_id = tokenizer_state["string_to_id"]
    tokenizer.id_to_string = {
        token_id: char
        for char, token_id in tokenizer.string_to_id.items()
    }
    tokenizer.next_id = model_config["vocab_size"]

    def tokenize_fn(text):
        ids = tokenizer.encode(text, model_config["max_len"])
        return torch.tensor(ids, dtype=torch.long)

    return model, tokenize_fn

        
        
        
    
        
        
