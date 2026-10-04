from .tokenizer import BPETokenizer
import torch
from pathlib import Path
from datetime import datetime
from .plot_training import save_training_history

ROOT = Path(__file__).resolve().parents[1]

  # 加载tokenizer
tokenizer = BPETokenizer.from_pretrained(
    ROOT / "ckpt" / "tokenizer.json"
)
# config
vocab_size = tokenizer.vocab_size
d_model=128
num_heads=4
num_layers=4
lr=3e-4
warmup_ratio=0.1
batch_size=32
dropout = 0.2
block_size = 256
epochs=15   
d_hidden = 4 * d_model
seed = 42
stride = 16
if torch.cuda.is_available() :
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    
    
import random
import numpy as np
import torch
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    # 遇到非确定性算子时发出警告
    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MY_Dataset(Dataset):
    def __init__(self , data , block_size , stride):
        self.stride = stride
        self.data = torch.as_tensor(data , dtype=torch.long)
        self.block_size = block_size
        
        
    def __len__(self):
        return max(
            0,
            (len(self.data) - self.block_size - 1)
            // self.stride + 1,
        )

    def __getitem__(self, idx):
        """根据索引返回单个样本（特征, 标签）"""
        idx = idx * self.stride
        sample = self.data[idx : idx + self.block_size]
        label = self.data[idx + 1 : idx + self.block_size + 1]
        return sample, label
    
import math
def create_warmup_cosine_scheduler(optimizer, total_steps, warmup_ratio):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_multiplier(current_step):
        # Phase 1: linearly increase from about 0 to the configured base LR.
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)

        # Phase 2: decay from the base LR to 0 following half a cosine wave.
        cosine_steps = max(1, total_steps - warmup_steps)
        progress = float(current_step - warmup_steps) / float(cosine_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_multiplier,
    )
    return scheduler, warmup_steps


def train_loop(
    dataloader,
    model,
    loss_fn,
    optimizer,
    scheduler,
    device,
):
    model.train()

    total_loss = 0.0
    total_tokens = 0

    for batch_idx, (X, y) in enumerate(dataloader):
        # X、y: (B, T)，标签比输入向后错一位
        X = X.to(device)
        y = y.to(device)

        # 清除上一步的梯度
        optimizer.zero_grad(set_to_none=True)

        # 训练时输入完整窗口，不使用 KV cache
        logits = model(X)  # (B, T, vocab_size)

        # 每个位置都预测下一个 token
        loss = loss_fn(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
        )

        # 计算梯度，并限制梯度范数
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        # 记录本次参数更新实际使用的学习率
        current_lr = optimizer.param_groups[0]["lr"]

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # 按 token 数统计平均损失
        num_tokens = y.numel()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

        if batch_idx % 100 == 0:
            print(
                f"batch: {batch_idx + 1}/{len(dataloader)}  "
                f"loss: {loss.item():.4f}  "
                f"lr: {current_lr:.2e}"
            )

    if total_tokens == 0:
        raise ValueError("训练 DataLoader 中没有样本")

    avg_loss = total_loss / total_tokens
    print(f"Train loss: {avg_loss:.4f}")

    return avg_loss
def test_loop(dataloader, model, loss_fn,device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    total_loss = 0.0
    total_tokens = 0

    model.eval()

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            loss = loss_fn(
                pred.reshape(-1, pred.shape[-1]),
                y.reshape(-1),
            )

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    print(f"Dev loss: {avg_loss:.4f}, PPL: {perplexity:.2f}")
    return avg_loss, perplexity

    
if __name__ == "__main__":
    
  
    # 加载数据集
    train_text = (ROOT / "data" / "train.txt").read_text(
        encoding="utf-8"
    )

    train_ids = torch.tensor(
        tokenizer.encode(train_text),
        dtype=torch.long,
    )
    
    dev_text = (ROOT / "data" / "dev.txt").read_text(
        encoding="utf-8"
    )

    dev_ids = torch.tensor(
        tokenizer.encode(dev_text),
        dtype=torch.long,
    )
    
    set_seed(seed=seed) 
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    train_dataset = MY_Dataset(train_ids, block_size , stride=stride)
    dev_dataset = MY_Dataset(dev_ids, block_size , stride= stride)

    # Dataloader
    train_dataloader = DataLoader(train_dataset , batch_size= batch_size , shuffle= True , generator= train_generator)
    dev_dataloader = DataLoader(dev_dataset , batch_size= block_size )
    
    model_config = {
            "vocab_size" : tokenizer.vocab_size,
            "d_model": d_model,
            "num_heads": num_heads,
            "d_hidden": d_hidden,
            "num_layers": num_layers,
            "block_size":block_size,
            "dropout": dropout
        }
    
    
    # model
    from .model import MiniGPT
    
    model = MiniGPT(**model_config)
    model.to(device)
    
    # train
    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr = lr, 
                                 weight_decay= 0.01
                                 )
    total_steps = epochs * len(train_dataloader)
    scheduler, warmup_steps = create_warmup_cosine_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_ratio=warmup_ratio,
    )
    print(
        f"total steps: {total_steps}, warmup steps: {warmup_steps}, "
        f"base lr: {lr:.2e}"
    )
    
    loss_fn = torch.nn.CrossEntropyLoss()
    best_dev_loss = float("inf")

    run_dir = ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    history = []
    run_config = {
        "model_config": model_config,
        "learning_rate": lr,
        "weight_decay": 0.01,
        "epochs": epochs,
        "batch_size": batch_size,
        "dev_batch_size": dev_dataloader.batch_size,
        "train_stride": train_dataset.stride,
        "dev_stride": dev_dataset.stride,
        "seed": seed,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "warmup_ratio": warmup_ratio,
        "device": str(device),
    }
    print(f"训练记录和曲线保存到：{run_dir}")
    
    patience = 5
    epochs_without_improvement = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}\n----------------------------")

        train_loss = train_loop(
            train_dataloader, model, loss_fn,
            optimizer, scheduler, device,
        )

        dev_loss, dev_ppl = test_loop(
            dev_dataloader, model, loss_fn, device,
        )

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss

            checkpoint = {
                "model_config": model_config,
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "best_dev_loss": best_dev_loss,
            }

            torch.save(checkpoint, ROOT / "ckpt" / "best.pt")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "dev_ppl": dev_ppl,
            "lr_next_step": optimizer.param_groups[0]["lr"],
        })
        save_training_history(history, run_config, run_dir)
        
        if epochs_without_improvement >= patience :
            print("!!!!!Early Stop!!!!!")
            break
            
        
        
        
