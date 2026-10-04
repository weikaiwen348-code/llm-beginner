## baseline：
config": {
    "model_config": {
      "vocab_size": 500,
      "d_model": 128,
      "num_heads": 4,
      "d_hidden": 512,
      "num_layers": 4,
      "block_size": 256,
      "dropout": 0
    },
    "learning_rate": 0.0003,
    "weight_decay": 0.01,
    "epochs": 40,
    "batch_size": 32,
    "dev_batch_size": 256,
    "train_stride": 16,
    "dev_stride": 16,
    "seed": 42,
    "total_steps": 1920,
    "warmup_steps": 192,
    "warmup_ratio": 0.1,
    "device": "mps"
  }

# 实验结果：
/Users/weikaiwen/llm-beginner/task-2-mini-gpt/runs/20260930_082040_995881
# 分析：
过拟合明显

## 实验一：
改动dropout = 0.1

# 实验结果：
最佳epoch还是8，但best dev PPL = 61.73 ，相比baseline有下降

## 实验二：
由于上轮dropout有效果，这轮加大dropout= 0.2
加入early——stop patience = 5

# 结果分析：
best-epoch = 9 有提升 best dev PPL = 60.90有提升，但不是很多

## 实验三：
num_layers = 2 减少参数，减轻过拟合

# 结果
best_epoch = 10 
best dev PPL = 64.23  反而上升，不太行

## 实验四
epochs = 15 缩短学习率
best_epoch = 11
best dev PPL = 60.13



## 实验五
vocab_size = 400
困惑度达到要求