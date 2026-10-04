import torch
from torch import nn
from .block import TransformerBlock
from pathlib import Path
from .tokenizer import BPETokenizer
from .sampling import sample_next_token
class MiniGPT(nn.Module) :
    def __init__(self,  vocab_size , d_model, num_heads ,d_hidden , num_layers ,block_size , dropout = 0):
        super().__init__()
        self.block_size = block_size        
        self.token_embedding = nn.Embedding(vocab_size , d_model)
        
        self.blocks = nn.ModuleList([TransformerBlock(d_model , num_heads , d_hidden , dropout) for _ in range(num_layers)])
        
        self.final_ln = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False,
        )
        
    def forward(self , ids , kv_cache=None, return_cache=False):
        
        T_new = ids.shape[1]

        if kv_cache is None:
            past_len = 0
        else:
            past_len = kv_cache[0][0].shape[-2]

        total_len = past_len + T_new
        
        if total_len > self.block_size:
            raise ValueError(
                f"sequence length {total_len} exceeds "
                f"block_size {self.block_size}"
            )
            
        x = self.token_embedding(ids)
        new_caches = []
        
        for i , block in enumerate(self.blocks) :

            if kv_cache is None:
                layer_cache = None
            else:
                layer_cache = kv_cache[i]

            if return_cache:
                x, layer_new_cache = block(
                    x,
                    kv_cache=layer_cache,
                    return_cache=True,
                )
                new_caches.append(layer_new_cache)

            else:
                x = block(
                    x,
                    kv_cache=layer_cache,
                    return_cache=False,
                )

        x = self.final_ln(x)

        logits = self.lm_head(x)

        if return_cache:
            return logits, new_caches

        return logits
            
    @torch.no_grad()
    def generate(self, prompt_ids , max_new_tokens , temperature = 0 , top_k = None , top_p = None) :
        
        # 完整生成结果
        generated = prompt_ids.clone()
        logits , caches = self.forward(prompt_ids , kv_cache= None , return_cache= True)
          
        
        for step in range(max_new_tokens):
            
            # logits[B , T , vocab_size] --> next_token_logits[B , vocab_size]
            # 最后一个位置预测下一个 token
            next_token_logits = logits[:, -1, :]

            # greedy
            next_token = sample_next_token(next_token_logits , temperature = temperature, top_k= top_k , top_p= top_p )

            # 保存生成出的 token
            generated = torch.cat(
                [generated, next_token],
                dim=1
            )
            if step == max_new_tokens - 1:
                break

            # 只把新 token 送进去
            logits, caches = self.forward(
                next_token,
                kv_cache=caches,
                return_cache=True
            )

        return generated


def load_for_eval(ckpt_path):
    ckpt_path = Path(ckpt_path)

    checkpoint = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=True,
    )

    # 按训练时的配置建立相同结构
    model = MiniGPT(**checkpoint["model_config"])

    # 加载训练好的参数
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 加载训练时配套使用的词表
    tokenizer = BPETokenizer.from_pretrained(
        ckpt_path.parent / "tokenizer.json"
    )

    return model, tokenizer