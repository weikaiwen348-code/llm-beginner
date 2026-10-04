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

## 2026-10-04：完整模型、训练评估与采样阶段总结

本节更新前一阶段的完成情况；上面的“下一步”保留为当时的学习记录。

### 已完成的模块

- **Tokenizer（M1）**：已生成正式词表，支持 JSON 保存、加载和中文 roundtrip；当前配套 tokenizer 的词表大小为 400。
- **完整模型（M2）**：实现 Pre-LN decoder block（attention、两层 GELU FFN、残差、dropout）和 `MiniGPT`，包括 embedding、多层 block、final LayerNorm 和 lm_head。
- **KV cache（M3）**：逐层传递缓存；生成时先处理完整 prompt，之后每次只 forward 新 token；`forward()` 通过 `block_size` 限制历史缓存与新增输入的总长度。
- **训练与评估（M4）**：实现错位一位的 next-token 数据窗口、独立 train/dev、AdamW、warmup + cosine、梯度裁剪、随机种子和按 dev loss 早停。记录每轮 loss/PPL，保存到独立 run 的 `history.json` 和 `training_curves.png`。
- **模型保存加载**：checkpoint 包含模型配置、参数、epoch 和最佳 dev loss；`load_for_eval()` 按配置重建模型、加载同目录 tokenizer，并切换到 eval 模式。
- **采样（M5，代码已接入）**：新增 `sample_next_token()`，支持 greedy、temperature、top-k、top-p 及组合；`generate()` 已接入采样函数，继续复用 KV cache。生成质量与不同策略的样例对比尚未完成。
- **PPL 自检**：按模型上下文长度分块，使用输入与右移标签计算 token 总 NLL，再除以被评分 token 数并取指数，避免一次输入超长序列。

### 本次修复的采样 bug

1. **循环导入**：`model.py` 导入 sampling，sampling 又导入 `MiniGPT`，导致类尚未定义就被访问。删除 sampling 对模型的反向依赖；采样只需要 logits，不需要模型或 tokenizer。
2. **零温度仍继续执行**：greedy 分支原先只赋值，随后仍除以 temperature。改成 `temperature == 0` 时直接返回 argmax。
3. **top-k 覆盖温度缩放**：原先从原始 logits 取 top-k，导致启用 top-k 后温度失效。改为从缩放后的 `filtered_logits` 取候选。
4. **采样函数缺少返回值**：原先抽样后没有 `return`，调用者得到 `None`；现在返回形状为 `(B, 1)` 的 token ID。

Top-p 的关键理解：按概率降序累计，保留第一个使累计概率跨过阈值的 token；其余候选填 `-inf`，重新 softmax 后抽样，而不是直接删除所有累计概率大于阈值的位置。

### 2026-10-04 实测结果

使用当前 `ckpt/best.pt` 与配套 tokenizer，在 CPU 上直接调用自检函数，没有重新训练或覆盖 `eval/result.json`。

| 检查项 | 结果 |
| --- | --- |
| Tokenizer roundtrip | 通过 |
| KV cache 等价性 | 通过，最大绝对误差 `3.337860107421875e-06`，小于 `1e-4` |
| 唐诗 dev PPL | `44.63`，低于阈值 `50`，评分 token 数为 `3075` |
| Greedy / top-k=1 | 人工 logits 测试均选中最大值所在 ID |
| Temperature + top-k | 实际送入抽样的概率与预期缩放、过滤结果一致 |
| Top-p 阈值边界 | 对概率 `[0.5, 0.3, 0.15, 0.05]`、`p=0.9`，正确保留前三项并重新归一化 |
| 生成调用冒烟测试 | greedy、温度采样、top-k、top-p、组合采样五种配置均能生成 4 个新 token，输出形状正确 |

采样检查为本次临时运行的小测试，尚未保存为仓库中的自动化测试。生成冒烟测试只验证调用与输出形状，不代表文本已经连贯或 UTF-8 一定合法。

### 实验结论与比较边界

- 已探索 dropout、层数、学习率调度周期和词表大小；更详细的历史实验见 `experiment.md` 与各 run 的 `history.json`。
- 500 词表、4 层、dropout=0.2、15 epoch 的训练验证最佳 PPL 约为 `60.13`；1000 词表实验的最佳训练验证 PPL 约为 `248.29`，后期验证 loss 进入平台期。
- 500→1000 词表时，训练 token 数从 `24551` 降到 `18855`，15 epoch 的计划更新次数从 `720` 降到 `555`。相同 epoch 并不等于相同更新预算。
- 当前 400 词表 checkpoint 的正式自检 PPL 达标，但不能仅凭 PPL 数字认定它比其他词表的文本建模更好：跨词表预测单位不同，应补充一致口径的 BPB 和生成样例比较。
- 训练验证使用重叠窗口，正式自检使用非重叠分块；两者 PPL 必须标注口径，不能混作同一个指标。

### 提交前状态与后续待办

- **完成状态**：M1–M4 已通过当前检查；M5 采样实现已接入，尚需保存同一 prompt、固定种子组、不同策略的生成样例，分析多样性、重复、连贯性和非法 UTF-8。
- **边界检查**：补充 temperature/top-k/top-p 参数校验，以及空 prompt、负生成长度和生成长度预算检查。生成时需显式使用 `model.eval()`；`no_grad()` 不会关闭 dropout。
- **实验归档**：当前权重仍写入公共 `ckpt/best.pt`，不同实验会覆盖；需把每个 run 对应的模型与 tokenizer 单独保存。现有 checkpoint 足够推理，但未保存优化器、调度器及随机状态，尚不支持完整恢复训练状态。
- **其他待办**：特殊 token（BOS/EOS/PAD）设计、生成解码容错、RoPE 频率表复用；weight tying、SwiGLU 可作为后续可选实验。
- 本次仅更新学习日志并运行诊断，未修改实现、暂存文件或执行 commit。`ckpt/`、数据和 `eval/result.json` 被任务目录的 `.gitignore` 忽略；提交时不要误把全部本地产物或 `.vscode/` 一并加入。
