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

## 2026-09-18：第一次完整分类训练

今天完成了字符级 tokenizer、Dataset、DataLoader 和第一版完整训练循环，将此前实现的 TransformerClassifier 真正用于 ChnSentiCorp 情感分类。

### 1. 实验配置

本次实验使用以下配置：

| 配置项 | 数值 |
|---|---:|
| 训练样本 | 9600 |
| 验证样本 | 1200 |
| 字符词表大小 | 4252 |
| `max_len` | 32 |
| `d_model` | 128 |
| attention heads | 4 |
| Transformer layers | 4 |
| FFN hidden size | 512 |
| batch size | 32 |
| learning rate | 3e-4 |
| epochs | 5 |

优化器使用 AdamW，`weight_decay=0.01`；学习率使用按 step 更新的 cosine schedule。模型根据 `pad_id` 自动构造 padding mask，并在 masked mean pooling 时再次排除 PAD。每轮训练后在 validation set 上计算 loss 和 accuracy，按 dev accuracy 保存 `ckpt/best.pt`，同时设置 `patience=2` 的 early stopping。

### 2. 实验结果

| Epoch | Train loss | Dev loss | Dev accuracy |
|---:|---:|---:|---:|
| 1 | 0.597036 | 0.526252 | 0.733333 |
| 2 | 0.497972 | 0.503737 | 0.752500 |
| 3 | 0.438644 | 0.485892 | 0.774167 |
| 4 | 0.389709 | **0.476567** | 0.780000 |
| 5 | 0.354509 | 0.482503 | **0.780833** |

训练 loss 持续下降，dev accuracy 从 0.7333 提升到 0.7808，说明 attention、padding mask、pooling、反向传播和优化循环整体能够正常工作。不过当前结果还没有达到任务要求的 0.80。

Epoch 5 的训练 loss 继续下降，但 dev loss 从 0.4766 回升至 0.4825，accuracy 只增加约 0.0008，说明模型开始出现轻微过拟合，并且接近当前配置的性能上限。因为每轮 accuracy 都有小幅提升，`patience=2` 的 early stopping 本次没有触发。

### 3. 发现的主要限制

当前最大的限制是 `max_len=32`。统计发现，训练集中约 94.1% 的文本长度超过 32，截断后只保留了全部字符的约 29.2%。评论后半部分可能包含转折后的关键情感信息，例如“但是很失望”，过短的输入会直接丢失这些信号。

| `max_len` | 被截断的训练样本 | 保留的总字符 |
|---:|---:|---:|
| 32 | 94.1% | 29.2% |
| 64 | 57.6% | 51.1% |
| 96 | 39.9% | 65.3% |
| 128 | 29.9% | 75.3% |

另外，当前 block 尚未使用 dropout；从 dev loss 的变化看，后续可以把 dropout 作为单独的对照实验。训练过程也还缺少 gradient clipping 和 warmup，但本次曲线较稳定，它们不是目前最明显的瓶颈。

### 4. Checkpoint 与评测接口

目前 `best.pt` 只保存了 `model.state_dict()`。它没有包含模型配置、字符词表、`pad_id`、`unk_id` 和 `max_len`，因此还不能由 `load_for_eval(ckpt_path)` 独立恢复。下一次正式训练前，需要先确定可复现的 checkpoint 格式，并让训练和评测共用同一份 tokenizer 实现。

### 5. 下一组实验计划

- [ ] 把 tokenizer 从 `train.py` 拆到独立模块，供训练和评测复用
- [ ] checkpoint 同时保存 state dict、模型配置和字符词表
- [ ] 实现 `load_for_eval(ckpt_path)`
- [ ] 单独将 `max_len` 提高到 96 或 128，保持其他参数不变进行对照
- [ ] 将最大 epochs 增加到 8，继续使用 `patience=2`
- [ ] 若仍出现过拟合，再单独加入 `dropout=0.1`
- [ ] 加入 gradient clipping，并记录是否影响训练稳定性

本次实验最重要的认识是：训练 loss 正常下降并不代表任务已经完成，还要同时观察 dev accuracy、dev loss、输入截断比例以及模型能否被完整恢复。实验配置本身也是模型能力的一部分。

## 2026-09-18：增加上下文长度并通过分类准确率门槛

第二次完整训练保持 `d_model=128`、4 个 head、4 层 Transformer、FFN hidden size 512、batch size 32 和学习率 3e-4 不变，只把 `max_len` 从 32 增加到 64。词表大小仍为 4252，训练 5 个 epoch。

| Epoch | Train loss | Dev loss | Dev accuracy |
|---:|---:|---:|---:|
| 1 | 0.556171 | 0.495631 | 0.775833 |
| 2 | 0.450619 | 0.449123 | 0.798333 |
| 3 | 0.387495 | **0.432703** | 0.805833 |
| 4 | 0.339790 | 0.451403 | 0.800833 |
| 5 | 0.312985 | 0.445003 | **0.812500** |

最佳 dev accuracy 达到 0.8125，超过任务要求的 0.80；`eval/run.py` 的分类准确率自检也得到 0.8125。与 `max_len=32` 的 0.7808 相比，本次提升约 3.17 个百分点，验证集中多预测正确约 38 条样本。结果支持“更多上下文能保留评论后半段情感信息”的判断，但由于两次实验没有固定随机种子，还不能把全部提升严格归因于序列长度；正式消融实验应使用相同 seed，最好重复多次报告均值。

Epoch 3 的 dev loss 最低，但 epoch 5 的 accuracy 最高，因此按任务规定的 dev accuracy 保存 epoch 5。后两轮 train loss 继续下降而 dev loss 回升，仍能看到轻微过拟合。后续可单独加入 dropout 做对照。

本轮同时完成了 tokenizer 模块拆分和新版 checkpoint：`best.pt` 现在包含模型参数、模型配置、字符词表、最佳准确率和 epoch，可以由 `load_for_eval` 在 CPU 上独立恢复。

### 注意力可视化计划

模型前向现已支持可选返回每一层的注意力矩阵，形状为 `(B, H, T, T)`。可视化脚本从验证集选择预测正确的正面、负面和长文本，默认绘制最后一层第 0 个 head；横轴表示被读取的 key 字符，纵轴表示发起查询的 query 字符。单个 head 的高权重只能表示信息混合比例，不能直接等价为特征的因果重要性，解读时还需要比较不同 layer/head 或结合遮挡实验。

实际生成的三张图保存在 `figures/{positive,negative,long}.png`。最后一层第 0 个 head 中，正面样本平均关注度较高的字符包括“就”“一”“成”“是”，并没有只集中在“不”“错”等直接情感字符；负面样本最关注“价”，与“降价了100元”的抱怨有关；长文本则较多关注“婚”“没”“必”等与“必须出示结婚证”语义相关的字符。这说明不同查询位置会共享少数关键 key，但一个 head 可能同时编码句法、位置或主题线索，不能仅凭一张热图断言模型的分类依据。

## 2026-09-18：`max_len=128` 与 warmup 实验

本轮把 `max_len` 从 64 提高到 128，同时使用 10% linear warmup + cosine decay。其余主要配置保持不变：`d_model=128`、4 个 head、4 层 Transformer、FFN hidden size 512、batch size 32、AdamW、基础学习率 3e-4，共训练 5 个 epoch。总训练步数为 1500，其中前 150 step 做线性 warmup。

| Epoch | Train loss | Dev loss | Dev accuracy |
|---:|---:|---:|---:|
| 1 | 0.556649 | 0.465762 | 0.792500 |
| 2 | 0.413418 | 0.384809 | 0.826667 |
| 3 | 0.341596 | 0.377511 | 0.825833 |
| 4 | 0.287118 | **0.356837** | 0.843333 |
| 5 | 0.256681 | 0.368073 | **0.846667** |

最佳 dev accuracy 达到 0.8467，即验证集 1200 条样本中预测正确 1016 条，距离参考基线 0.85 只差 4 条样本。checkpoint 元数据确认最佳模型来自 epoch 5，并完整记录了 `max_len=128`、warmup ratio 0.1、150 个 warmup steps 和 1500 个 total steps。

与同样采用 10% warmup 的 `max_len=64` 实验相比，最佳准确率从 0.8092 提高到 0.8467，增加 3.75 个百分点，即多预测正确约 45 条样本。`max_len=64` 只能保留约 51.1% 的训练字符，而 128 能保留约 75.3%，结果进一步支持“长评论的后半段包含重要情感信息”这一判断。由于这些实验仍未固定随机种子，这个提升不能被视为严格的单变量消融结论，但幅度已经明显大于几条样本的普通波动。

训练 loss 五轮持续下降；dev loss 在 epoch 4 达到最低值，epoch 5 回升，但 accuracy 仍从 0.8433 提升到 0.8467。这说明模型在第五轮的预测置信度可能变得更极端：少数错例带来更大的交叉熵损失，同时分类正确的样本数量略有增加。因为任务按 dev accuracy 选择 checkpoint，所以保存 epoch 5 是正确的；如果目标是概率校准或最低验证损失，则会选择 epoch 4。

下一步应先固定随机种子，再对 `max_len=64/96/128` 做可重复对照。若继续提升分类效果，可以在 `max_len=128` 上加入 dropout 以缓解后期 dev loss 回升；若优先完成加分实验，则可以保持当前最佳配置，比较 1、2、4、8 个 attention heads。

## 2026-09-18：Dropout 强度与训练时长实验

本轮在 `max_len=128` 的模型上加入 dropout，并固定 `seed=42`。Dropout 用在 embedding 与位置编码相加之后、FFN 中间，以及 attention/FFN 输出进入 residual 之前。其他主要配置保持为 `d_model=128`、4 个 head、4 层、batch size 32、AdamW、学习率 3e-4 和 10% warmup。

先用 5 个 epoch 比较 dropout 强度：

| Dropout | Epochs | 最终 Train loss | Dev loss | Dev accuracy |
|---:|---:|---:|---:|---:|
| 0.1 | 5 | 0.333257 | **0.393554** | **0.834167** |
| 0.2 | 5 | 0.374522 | 0.398370 | 0.826667 |

在相同训练轮数下，`dropout=0.1` 的训练 loss 更低，dev loss 和 accuracy 也都优于 0.2。`dropout=0.2` 的正则化更强，使模型在五轮内仍有明显欠拟合或尚未充分收敛，因此当前任务更适合从 0.1 开始。

随后把 `dropout=0.1` 的最大训练轮数增加到 10。此时 cosine schedule 的总步数从 1500 增加到 3000，warmup steps 也从 150 增加到 300。已记录的后四轮结果如下：

| Epoch | Train loss | Dev loss | Dev accuracy |
|---:|---:|---:|---:|
| 7 | 0.247745 | **0.368047** | 0.853333 |
| 8 | 0.225826 | 0.385057 | **0.860000** |
| 9 | 0.212308 | 0.384932 | 0.859167 |
| 10 | 0.200056 | 0.388405 | **0.860000** |

当前最佳 checkpoint 为 epoch 8，dev accuracy 达到 0.86，即 1200 条验证样本中预测正确 1032 条，超过 0.85 参考基线。Epoch 10 虽然同样达到 0.86，但保存条件使用严格的 `>`，相等时不会覆盖已有最佳模型。Epoch 7 的 dev loss 最低，而 epoch 8 的分类正确数更多，说明后期模型在少数错例上的置信度变得更极端；本任务以 accuracy 为准，因此选择 epoch 8。

从五轮的 0.8342 到十轮的 0.86，说明 dropout 降低了前期拟合速度，但允许模型在更长训练中继续改善泛化。不过这两次运行不仅 epochs 不同，cosine 的总长度和 warmup steps 也同时改变，所以不能把提升全部归因于“多训练五轮”。若要严格比较 dropout，下一步应让 `dropout=0/0.1/0.2` 都使用相同的 seed、10 epochs、3000 total steps 和 300 warmup steps，再比较各自的最佳 epoch 与平均结果。



## 2026-09-18：attention heads 实验
# num_heads = 1
Epoch 10
-------------------------------
loss: 0.094928  [   32/ 9600]  lr: 8.99e-06
loss: 0.110824  [ 3232/ 9600]  lr: 4.00e-06
loss: 0.199558  [ 6432/ 9600]  lr: 9.94e-07
Avg loss: 0.210062 

Test Error: 
 Accuracy: 0.862500, Avg loss: 0.372619 


# num_heads = 2
Epoch 9
-------------------------------
loss: 0.316510  [   32/ 9600]  lr: 3.50e-05
loss: 0.282349  [ 3232/ 9600]  lr: 2.46e-05
loss: 0.260164  [ 6432/ 9600]  lr: 1.59e-05
Avg loss: 0.218532 

Test Error: 
 Accuracy: 0.854167, Avg loss: 0.376392 

# num_heads = 4
Epoch 10
-------------------------------
loss: 0.121693  [   32/ 9600]  lr: 8.99e-06
loss: 0.100376  [ 3232/ 9600]  lr: 4.00e-06
loss: 0.275710  [ 6432/ 9600]  lr: 9.94e-07
Avg loss: 0.200056 

Test Error: 
 Accuracy: 0.860000, Avg loss: 0.388405 

# num_heads = 6
Epoch 10
-------------------------------
loss: 0.123358  [   32/ 9600]  lr: 8.99e-06
loss: 0.148505  [ 3232/ 9600]  lr: 4.00e-06
loss: 0.178051  [ 6432/ 9600]  lr: 9.94e-07
Avg loss: 0.193010 

Test Error: 
 Accuracy: 0.862500, Avg loss: 0.373479 

# num_heads = 8
Epoch 8
-------------------------------
loss: 0.173323  [   32/ 9600]  lr: 7.48e-05
loss: 0.226764  [ 3232/ 9600]  lr: 6.03e-05
loss: 0.161016  [ 6432/ 9600]  lr: 4.69e-05
Avg loss: 0.220215 

Test Error: 
 Accuracy: 0.867500, Avg loss: 0.379856 

 并没有体现出单调的提升，还是采用head = 4 吧比较平均，计算速度快
 ## post-ln --》 pre-ln
 Epoch 10
-------------------------------
loss: 0.073506  [   32/ 9600]  lr: 8.99e-06
loss: 0.186173  [ 3232/ 9600]  lr: 4.00e-06
loss: 0.210196  [ 6432/ 9600]  lr: 9.94e-07
Avg loss: 0.220820 

Test Error: 
 Accuracy: 0.860833, Avg loss: 0.382834 
 也并没有明显提升

## 2026-09-18：Pre-LN 修复与 RoPE 实验

本轮尝试将可学习的绝对位置嵌入替换为 RoPE，并使用 Pre-LN Transformer block。主要训练配置为：`max_len=128`、`d_model=128`、4 个 head、4 层、FFN hidden size 512、`dropout=0.1`、batch size 32、AdamW、学习率 3e-4、10% warmup、最多 10 epochs，并固定 `seed=42`。

### 调试过程与 bug 记录

第一次实现中虽然创建了 `self.rope = RotaryEmbedding(self.depth)`，但在 attention 的 `forward` 中没有真正调用它。同时 `model.py` 已删除绝对位置嵌入，因此该次运行实际是“无位置编码”对照，而不是 RoPE：

| 实际模型 | Train loss | Dev loss | Dev accuracy |
|---|---:|---:|---:|
| 无位置编码 | 0.192350 | 0.394424 | 0.856667 |

修复方式是在 Q/K/V 拆分成多头后、计算 attention score 之前，对 Q 和 K 执行旋转：

```python
Q = self.split_heads(Q)
K = self.split_heads(K)
V = self.split_heads(V)

Q = self.rope(Q)
K = self.rope(K)
```

V 不做旋转。RoPE 使 Q/K 的点积包含相对位置信息，不需要再将位置向量加到 token embedding 上。

同时还发现了 Pre-LN 残差路径的 bug。错误版本先用 `x = ln(x)` 覆盖了原始残差流，然后再执行残差相加；正确的 Pre-LN 应保留未归一化的 `x`：

```python
attention_input = ln1(x)
x = x + attention(attention_input)

ffn_input = ln2(x)
x = x + ffn(ffn_input)
```

在所有 Transformer blocks 之后又增加了一个最终 LayerNorm，再进行 masked mean pooling。

### 修复后结果

checkpoint 元数据确认最佳模型来自 epoch 6：

| Epoch | Train loss | Dev loss | Dev accuracy |
|---:|---:|---:|---:|
| 6 | 0.187699 | **0.319983** | **0.890000** |

修复后模型在 1200 条验证样本中预测正确 1068 条。与先前 0.86 的最佳结果相比，提高了 3 个百分点，多预测正确 36 条。不过本轮同时修复了 Pre-LN 并加入 RoPE，因此还不能把全部提升都归因于 RoPE。严格消融应在相同的正确 Pre-LN 结构上，分别训练“绝对 PE”和“RoPE”，并保持其他超参数与随机种子不变。

这次调试说明：只在 `__init__` 中声明模块不代表它已参与前向计算；检查新结构时应沿完整数据流确认它确实影响了输出。后续还应删除 block 中未使用的 `ln3`，并在 checkpoint 配置中记录 `position_encoding` 和 `norm_type`。
