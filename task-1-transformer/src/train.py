import math

import torch
from torch import nn
from .model import TransformerClassifier
from .tokenizer import Tokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# 固定随机种子
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

class SentimentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx]["text"]
        label = self.dataframe.iloc[idx]["label"]

        ids = torch.tensor(
            self.tokenizer.encode(text, self.max_len),
            dtype=torch.long
            )

        label = torch.tensor(label, dtype=torch.long)
        return ids, label


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


def train_loop(dataloader, model, loss_fn, optimizer, scheduler,device):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    train_loss = 0.0
    num_batches = len(dataloader)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  "
                f"lr: {current_lr:.2e}"
            )
    train_loss /= num_batches
    print(f"Avg loss: {train_loss:>8f} \n")
    return train_loss

def test_loop(dataloader, model, loss_fn,device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(correct):>4f}, Avg loss: {test_loss:>8f} \n")
    return test_loss ,correct



from pathlib import Path
from functools import partial

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

device = torch.device("mps")


if __name__ == "__main__":
    
    train_df = pd.read_parquet(ROOT / "data" / "train.parquet")
    dev_df = pd.read_parquet(ROOT / "data" / "validation.parquet")

    print("train_df.shape:   ",train_df.shape)
    
    # config
    d_model=128
    num_heads=4
    num_layers=4
    lr=3e-4
    warmup_ratio=0.1
    batch_size=32
    dropout = 0.1
    epochs=10
    
    max_len = 128
       
    d_hidden = 4 * d_model
        
    pad_id=1
    seed = 42
    
    set_seed(seed=seed)
    
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    
    # 建词表
    tokenizer = Tokenizer()
    train_texts = train_df["text"].astype(str).tolist()
    tokenizer.build_vocab(train_texts)

    vocab_size = tokenizer.get_vocab_size()

    print("vocab size:", vocab_size)
    
    # datasets
    train_dataset = SentimentDataset(train_df, tokenizer, max_len)
    dev_dataset = SentimentDataset(dev_df, tokenizer, max_len)

    # dataloaders
    train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    generator=train_generator
    )

    dev_loader = DataLoader(
    dev_dataset,
    batch_size=batch_size,
    shuffle=False,
    )
    
    
    # Keep the constructor arguments together so the exact same architecture
    # can be rebuilt later by load_for_eval().
    model_config = {
        "vocab_size": vocab_size,
        "max_len": max_len,
        "d_model": d_model,
        "num_heads": num_heads,
        "d_hidden": d_hidden,
        "num_layers": num_layers,
        "pad_id": pad_id,
        "dropout": dropout
    }

    # model
    model = TransformerClassifier(**model_config)
    model.to(device)
    # train
    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr = lr, 
                                 weight_decay= 0.01
                                 )
    total_steps = epochs * len(train_loader)
    scheduler, warmup_steps = create_warmup_cosine_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_ratio=warmup_ratio,
    )
    print(
        f"total steps: {total_steps}, warmup steps: {warmup_steps}, "
        f"base lr: {lr:.2e}"
    )
    
    best_dev_acc = 0.0
    epochs_without_improvement = 0
    patience = 2
    
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        
        
        print(f"Epoch {epoch+1}\n-------------------------------")
    
        train_loss = train_loop(train_loader, model, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler,device=device)
        test_loss , test_acc= test_loop(dev_loader, model, loss_fn= loss_fn,device=device)
        
        
        if test_acc > best_dev_acc:
            best_dev_acc = test_acc
            epochs_without_improvement = 0

            ckpt_dir = ROOT / "ckpt"
            ckpt_dir.mkdir(exist_ok=True)

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": model_config,
                "tokenizer": {
                    "string_to_id": tokenizer.string_to_id,
                    "pad_id": tokenizer.pad_id,
                    "unk_id": tokenizer.unk_id,
                },
                "best_dev_acc": best_dev_acc,
                "epoch": epoch + 1,
                "training_config": {
                    "learning_rate": lr,
                    "warmup_ratio": warmup_ratio,
                    "warmup_steps": warmup_steps,
                    "total_steps": total_steps,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "seed": seed
                },
            }

            torch.save(checkpoint, ckpt_dir / "best.pt")
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print("Early stopping")
            break
        
        
        
        
        
    
    
