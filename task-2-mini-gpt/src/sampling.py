
import torch

from pathlib import Path


def sample_next_token(
    logits,             # (B, vocab_size)
    temperature=1.0,
    top_k=None,
    top_p=None,
):
    # logits[B , vocab_size]
    if temperature == 0:
        return logits.argmax(dim = -1 , keepdim = True)
    
    filtered_logits = logits / temperature
    
    
    if top_k is not None:
        values, indices = torch.topk(filtered_logits, k=top_k, dim=-1)
        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(
            dim=-1,
            index=indices,
            src=values
            )
        filtered_logits = filtered
        
    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(
                                                    filtered_logits,
                                                    descending=True,
                                                    dim=-1
                                                )
        sorted_probs = torch.softmax(
                                        sorted_logits,
                                        dim=-1
                                    )
        cumulative_probs = torch.cumsum(
                                            sorted_probs,
                                            dim=-1
                                        )
        sorted_remove = cumulative_probs > top_p
        sorted_remove[..., 1:] = (
        sorted_remove[..., :-1].clone()
    )

        sorted_remove[..., 0] = False
        remove_mask = torch.zeros_like(
                                            filtered_logits,
                                            dtype=torch.bool
                                        )

        remove_mask.scatter_(
                                dim=-1,
                                index=sorted_indices,
                                src=sorted_remove
                            )
        filtered_logits = filtered_logits.masked_fill(
                                                        remove_mask,
                                                        float("-inf")
                                                    )
        
        
    probs = torch.softmax(filtered_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token
    
    
        

    