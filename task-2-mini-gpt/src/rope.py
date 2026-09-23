import torch


def precompute_freqs_cis(dim: int, seq_len: int, position_offset=0, theta: float = 10000.0 , device = None):
    assert dim % 2 == 0
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度\theta_i
    freqs = 1.0 / (theta ** (torch.arange(0 , dim ,2 ,device=device,
    dtype=torch.float32).float() / dim))
    
    # 生成 token 序列索引
    t = torch.arange(
        position_offset,
        position_offset + seq_len,
        device=device,
    )
    
    # freqs.shape = [seq_len, dim // 2] 
    freqs = torch.outer(t, freqs).float()  # 计算m * \theta
    
    # 计算结果是个复数向量
    # 假设 freqs = [x, y]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) 
    return freqs_cis
    
    
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) :
    # xq.shape = [batch_size, num_heads , seq_len, dim]
    # xq_.shape = [batch_size, num_heads , seq_len, dim // 2, 2]
    
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    # 转为复数域
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)
    
    # 应用旋转操作，然后将结果转回实数域
    # xq_out.shape = [batch_size, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def rope(xq: torch.Tensor,
    xk: torch.Tensor,
    position_offset = 0):
    
    seq_len = xq.shape[-2]
    dim = xq.shape[-1]
    freqs_cis = precompute_freqs_cis(dim= dim , seq_len= seq_len, position_offset= position_offset, device = xq.device)
    
    xq_ , xk_ = apply_rotary_emb(xq = xq, xk = xk, freqs_cis = freqs_cis)
    return xq_, xk_
    
    
if __name__ == "__main__":
    B, H, T, D = 2, 4, 8, 32

    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)

    q_rotated, k_rotated = rope(q, k)

    print("q:", q.shape)
    print("q_rotated:", q_rotated.shape)
    print("k_rotated:", k_rotated.shape)

    assert q_rotated.shape == q.shape
    assert k_rotated.shape == k.shape

    # 旋转不应改变每一对向量的整体模长
    assert torch.allclose(
        q.norm(dim=-1),
        q_rotated.norm(dim=-1),
        atol=1e-5,
    )
    