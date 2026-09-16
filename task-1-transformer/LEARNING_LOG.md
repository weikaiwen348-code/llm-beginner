# Transformer 学习记录

## 2026-09-16：从 Attention 到 TransformerClassifier

今天主要完成了 `task-1-transformer` 中从注意力模块到分类模型主干的搭建。相比之前主要是“看懂代码”，今天开始真正从需求出发设计类、函数接口、输入输出和 Tensor shape。

### 1. Scaled Dot-Product Attention

在 `src/attention.py` 中实现并整理了缩放点积注意力：

- 输入 `Q/K/V` 形状：`(B, H, T, d_k)`
- 计算 `Q @ K^T` 得到 attention scores：`(B, H, T, T)`
- 使用 `sqrt(d_k)` 进行缩放
- 使用 mask 将被屏蔽位置填为 `-inf`
- 在最后一维执行 softmax
- 与 `V` 相乘得到 attention output

进一步理解了两类 mask：

- **padding mask**：用于屏蔽 `[PAD]`，模型中常整理为 `(B, 1, 1, T)`
- **causal mask**：用于禁止看到未来 token，常为 `(T, T)`，依靠 broadcasting 与 scores 对齐

关键认识：`scaled_dot_product_attention` 不应该假设 mask 一定是 padding mask；只需要接受能够广播到 scores 的 mask。

### 2. Multi-Head Attention

实现了 `MultiHeadAttention`：

- `Wq/Wk/Wv`：`d_model -> d_model`
- 将 `(B, T, d_model)` 拆成 `(B, H, T, d_k)`
- 每个 head 独立进行 scaled dot-product attention
- 将多头结果重新合并为 `(B, T, d_model)`
- 使用 `W_out` 做最终投影

进一步熟悉了：

- `view / transpose / contiguous`
- `d_k = d_model / num_heads`
- 多头 attention 的 shape 变化

### 3. 手写 LayerNorm 与 TransformerBlock

在 `src/block.py` 中手写了 LayerNorm：

- 沿最后一维计算 mean 和 variance
- 使用 `unbiased=False`
- 实现 `(x - mean) / sqrt(var + eps)`
- 使用可学习参数 `gamma` 和 `beta`

随后搭建 Post-LN TransformerBlock：

```text
x
 -> Attention
 -> Residual Add
 -> LayerNorm
 -> FFN
 -> Residual Add
 -> LayerNorm
 -> output
```

FFN 使用：

```text
d_model -> d_hidden -> GELU -> d_model
```

并理解了：

- `nn.Sequential` 适合固定流水线
- `nn.ModuleList` 负责注册多个模块，但 forward 需要自己遍历
- residual connection 本质是 Tensor 加法，不需要单独定义一个层

### 4. TransformerClassifier 主干

在 `src/model.py` 中完成了分类器主体：

```text
input_ids (B,T)
 -> Token Embedding
 -> + Position Embedding
 -> TransformerBlock x N
 -> Masked Mean Pooling
 -> Linear Classification Head
 -> logits (B,2)
```

具体实现包括：

- `nn.Embedding(vocab_size, d_model)` token embedding
- `nn.Embedding(max_len, d_model)` learned position embedding
- 使用 `num_layers` 控制 Block 数量
- 使用 `nn.ModuleList` 保存多个独立 TransformerBlock
- 根据 `pad_id` 自动生成 padding mask
- 对超过 `max_len` 的输入沿 token 维截断
- attention 使用 `(B,1,1,T)` mask
- mean pooling 使用 `(B,T,1)` valid mask，排除 PAD token
- 分类 head 将 `(B,d_model)` 映射为 `(B,2)` logits

### 5. 今天补上的工程知识

除了 Transformer 本身，还补了很多实际写代码时需要的知识：

- `super().__init__()`：初始化父类 `nn.Module`
- Python 模块导入：`from .attention import MultiHeadAttention`
- `self.xxx` 与局部变量的区别
- `nn.Parameter` 的作用
- `unsqueeze`、`sum(dim=...)`、broadcasting、`clamp(min=1)`
- batch 里的文本长度不同，需要 padding / truncation / mask
- `input_ids` 是 tokenizer 输出的整数 token id，而 embedding 将其映射为连续向量

### 6. 当前进度

已完成或基本完成：

- [x] Scaled Dot-Product Attention
- [x] Multi-Head Attention
- [x] LayerNorm
- [x] TransformerBlock
- [x] TransformerClassifier 主干
- [x] padding-aware mean pooling
- [x] `model(input_ids)` 时自动生成 padding mask

下一步：

- [ ] 确定 tokenizer / vocabulary / `pad_id`
- [ ] 编写 `train.py`
- [ ] AdamW + cosine LR schedule
- [ ] dev accuracy + early stopping
- [ ] 保存 `ckpt/best.pt`
- [ ] 实现 `load_for_eval(ckpt_path)`
- [ ] 生成并提交 `eval/result.json`
- [ ] 后续加入 dropout、attention weights 与热图

### 7. 今日最重要的理解

今天最大的变化不是“多写了几个类”，而是开始建立从数学定义到程序结构的映射：

```text
任务需求
 -> 模块职责
 -> 输入 / 输出
 -> Tensor shape
 -> 类和函数接口
 -> PyTorch 实现
```

具体 API 可以查询，但应该优先自己判断：数据现在是什么 shape、下一步想得到什么 shape、这个模块在整个模型中负责什么。
