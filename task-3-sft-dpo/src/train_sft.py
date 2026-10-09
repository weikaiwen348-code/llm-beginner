import json
import zipfile
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from .chat import format_messages, build_labels, tokenizer
from itertools import islice
from transformers import AutoModelForCausalLM
from .lora import inject_lora , get_lora_state
import sys
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "moss-sft" / "moss-003-sft-no-tools.jsonl.zip"
import random
import numpy as np

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

def moss_to_messages(record):
    # 助手设定放在最前面的 system 消息中
    messages = [
        {
            "role": "system",
            "content": record["meta_instruction"],
        }
    ]

    # 按第 1、2、3……轮依次读取
    for i in range(1, record["num_turns"] + 1):
        turn = record["chat"][f"turn_{i}"]

        user_text = (
            turn["Human"]
            .removeprefix("<|Human|>:")
            .rsplit("<eoh>", 1)[0]
            .strip()
        )

        assistant_text = (
            turn["MOSS"]
            .removeprefix("<|MOSS|>:")
            .rsplit("<eom>", 1)[0]
            .strip()
        )

        messages.append({
            "role": "user",
            "content": user_text,
        })
        messages.append({
            "role": "assistant",
            "content": assistant_text,
        })

    return messages

class SFTDataset(Dataset):
    def __init__(self, records, max_length=512):
        self.samples = []

        for record in records:
            messages = moss_to_messages(record) # 去moss格式
            text = format_messages(messages) #换成qwen格式

            encoded = tokenizer(
                text,
                add_special_tokens=False,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"][0]
            labels = build_labels(input_ids, messages)

            # 先在完整对话上制作 labels，再一起截断
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]

            # 截断可能只留下 system/user，这种样本无法训练回答
            if not (labels[1:] != -100).any():
                continue

            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "labels": labels,
            })

        if not self.samples:
            raise ValueError("截断后没有包含有效回答 token 的样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate_fn(batch): # 传入一个batch的数据，把他们补齐长度
   
    input_ids = pad_sequence(
        [sample["input_ids"] for sample in batch],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )

    attention_mask = pad_sequence(
        [sample["attention_mask"] for sample in batch],
        batch_first=True,
        padding_value=0,
    )

    labels = pad_sequence(
        [sample["labels"] for sample in batch],
        batch_first=True,
        padding_value=-100,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    
if torch.cuda.is_available() :
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    
def train_loop(model , optimizer , dataloader , max_norm = 1.0):
    model.train()
    
    total_loss = 0.0
    total_tokens = 0
    for batch_idx , batch in enumerate(dataloader):
        
        # 将数据转到设备
        for key , value in batch.items():
                batch[key] = value.to(device)
                
        # 清除上一步的梯度
        optimizer.zero_grad(set_to_none=True)
        
        # 训练时不需要cache
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            use_cache=False,
        )   
        logits = outputs.logits
        loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm,
                )
        
        optimizer.step()
        
        if batch_idx % 10 == 0:
                    print(
                        f"batch: {batch_idx + 1}/{len(dataloader)}  "
                        f"loss: {loss.item():.4f}  "
                    )
                    print()
        
        num_tokens = (batch["labels"][:, 1:] != -100).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
    
    train_loss = total_loss / total_tokens
    return train_loss

def val_loop(model, dataloader ):
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for batch_idx , batch in enumerate(dataloader):
                
            # 将数据转到设备
            for key , value in batch.items():
                batch[key] = value.to(device)
                
            outputs = model(**batch , use_cache = False)
            loss = outputs.loss
            
            num_tokens = (batch["labels"][:, 1:] != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            
        val_loss = total_loss / total_tokens
        return val_loss


def split_records(records, seed , val_ratio = 0.1):
#输入：原始对话列表、验证比例、随机种子
#输出：train_records、val_records 两个列表
    # 1. 浅拷贝一份列表，避免原地修改原数据
    shuffled = records.copy()

    # 2. 设置随机种子保证可复现性，并就地打乱列表
    random.seed(seed)
    random.shuffle(shuffled)

    # 3. 计算验证集的样本数量
    val_size = int(len(shuffled) * val_ratio)
    
    # 4. 切片拆分为验证集和训练集
    val_records = shuffled[:val_size]
    train_records = shuffled[val_size:]

    return train_records, val_records

def test_model(model , dataloader ):
    val_loss = val_loop(model , dataloader)
    print("源模型平均loss ： " , val_loss)
    
    
if __name__ == "__main__":
    
    seed = 40
    batch_size = 4
    records = []
    val_ratio = 0.1
    MAX_SAMPLES = 100
    
    lora_config = {
    "base_model": "Qwen2.5-0.5B",
    "r": 8,
    "alpha": 16,
    "target_modules": ["q_proj", "v_proj"],
}

    data_config = {
    "seed": seed,
    "max_samples": MAX_SAMPLES,
    "val_ratio": val_ratio,
}
    set_seed(seed = seed)
    
    # 打开 ZIP，再打开其中的 JSONL 文件
    with zipfile.ZipFile(DATA_PATH) as archive:
        with archive.open("moss-003-sft-no-tools.jsonl") as file:
            for line in islice(file, MAX_SAMPLES):
                records.append(json.loads(line))
    
    train_records , val_records = split_records(records , seed , val_ratio= val_ratio)
    # 读取前 个对话
    train_dataset = SFTDataset(train_records)
    val_dataset = SFTDataset(val_records)

    # 验证集shuffle = false
    train_dataloader = DataLoader(train_dataset , batch_size= batch_size, shuffle= True , collate_fn= collate_fn)
    val_dataloader = DataLoader(val_dataset , batch_size= batch_size , shuffle= False , collate_fn= collate_fn)
    
    
    model = AutoModelForCausalLM.from_pretrained(
    ROOT / "models" / "Qwen2.5-0.5B",
    local_files_only=True,
)
    model.to(device)
    model = inject_lora(model , 
    target_modules=lora_config["target_modules"],
    r=lora_config["r"],
    alpha=lora_config["alpha"],)
    
    
    # 过滤掉 requires_grad 为 False 的参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    
    epochs = 10
    
    test_model(model , val_dataloader)
    sys.exit(0)
    
    
    check_point = {      
    }
    best_loss = float("inf")
    save_dir = ROOT / "ckpt" / "sft"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    

    # 每次训练使用新的日志文件，保留以前的实验记录。
    log_dir = ROOT / "runs" / "sft"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_number = 1
    history_path = log_dir / f"history_{run_number:03d}.json"
    while history_path.exists():
        run_number += 1
        history_path = log_dir / f"history_{run_number:03d}.json"
    history = []
    print(f"Loss history: {history_path}")
    
    

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}\n----------------------------")
        
        
        train_loss = train_loop(model , optimizer ,train_dataloader)
        val_loss = val_loop(model , val_dataloader)
        print ("train_loss : " , train_loss)
        print("val_loss :   " , val_loss)

        # 每轮都记录，包括验证 loss 没有改善的轮次。
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })
        with history_path.open("w", encoding="utf-8") as file:
            json.dump({
                "lora_config": lora_config,
                "data_config": data_config,
                "metrics": history,
            }, file, ensure_ascii=False, indent=2)
        
        if val_loss <= best_loss :
            best_loss = val_loss
            
            check_point = {
                "epoch" : epoch + 1,
                "lora_state" : get_lora_state(model),
                "optimizer" : optimizer.state_dict(),
                "train_loss" : train_loss,
                "val_loss" : val_loss,
                "lora_config": lora_config,
                "data_config": data_config,
            }
            
            torch.save(check_point, save_dir / f"epoch_{epoch + 1}.pt")
            
        print()
        
        
        
        
        
        

    
