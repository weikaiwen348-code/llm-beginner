# Task 2 学习日志

## 2026-09-23：BPE、RoPE 与 KV Cache Attention

今天完成了 mini-GPT 的前两个核心模块。

### Byte-level BPE Tokenizer

- 从 256 个基础 byte token 出发，统计相邻 token pair 频率并迭代合并。
- 实现了 `train`、`encode`、`decode`、`vocab_size`、`save` 和 `from_pretrained`。
- JSON 中按训练顺序保存 `[left_id, right_id, new_id]` merge 规则；加载时从固定的 256 个 byte token 重建完整词表。
- 使用 500 大小词表测试时，加载前后编码结果一致；“床前明月光”、`Hello, world!` 和“深度学习需要数学基础”均能正确 roundtrip。
- 训练语料从 43,292 个 UTF-8 bytes 压缩为 24,551 个 BPE tokens。

当前 tokenizer 核心逻辑已完成，但尚未生成正式的 `ckpt/tokenizer.json`，也尚未运行 M1 自检。

### RoPE

- 用复数旋转实现 RoPE，将 head dimension 中的相邻两维配成一组。
- RoPE 只作用于 Q/K，不作用于 V。
- 增加 `position_offset`，使增量解码的新 token 能继续使用历史位置编号。
- 修正了 `float()` 调用、`flatten(-2)` 维度和新建张量的 device 一致性。
- CPU 测试确认 RoPE 前后形状不变，且向量模长保持不变。

### Causal Attention 与 KV Cache

- 实现了 causal multi-head attention，将未来位置的 attention score 填为 `-inf`。
- 修正了 mask 方向：使用主对角线以上为 `True` 的屏蔽矩阵，当前 token 仍可以看到自己。
- KV cache 使用 `(past_k, past_v)`，并沿序列维 `-2` 追加新 K/V。缓存中的 K 已经应用 RoPE，历史 K 不会被重复旋转。
- 通过 `past_len` 为新 Q/K 设置 RoPE offset，并用 query/key 的绝对位置构造 `(T_new, T_total)` 矩形 causal mask。
- 实测 cache 长度能从 1 逐步增长到 7，最终 K/V 形状均为 `(2, 4, 7, 32)`。
- 完整前向与逐 token KV cache 前向的最大误差为 `2.98e-7`，低于任务要求的 `1e-4`。
- 修改未来 token 后，过去位置输出的最大变化为 `0.0`，证明 causal mask 有效。

### 今天遇到的关键 bug

1. `masked_fill` 的返回值没有赋回 `scores`，mask 实际未生效。
2. 使用包含主对角线的上三角 mask，导致第一个 query 的所有 score 都是 `-inf`，softmax 产生 NaN。
3. 用 `x[:, past_len, :]` 取新 token，意外删除了序列维。正确契约是调用者只传入本轮新 token，attention 内部直接投影全部 `x`。
4. cache 场景下 Q 和 K 的序列长度不同，固定 `T x T` mask 需要改为 `(T_new, T_total)`。

### 下一步

1. 生成 `ckpt/tokenizer.json` 并运行 tokenizer roundtrip 自检。
2. 实现 Pre-LN decoder block，将每层的 attention cache 向上传递。
3. 实现 `MiniGPT`，用 list 管理所有层的 KV cache。
4. 在本机 MPS 上检查复数 RoPE 的兼容性；如果 MPS 不支持相关复数操作，改用纯实数 `sin/cos` 版本。
