from pathlib import Path
from transformers import AutoTokenizer
import torch

# chat.py 在 src/ 下，往上一层就是 task-3-sft-dpo
ROOT = Path(__file__).resolve().parents[1]


# 加载配套的分词器
tokenizer = AutoTokenizer.from_pretrained(
    ROOT / "models" / "Qwen2.5-0.5B",
    local_files_only=True,
)


def format_messages(messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return text
    
def build_labels(input_ids, messages):
    # 1. 使用同一份完整文本，取得每个 token 的字符位置
    text = format_messages(messages)

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    # 确认传入的 IDs 确实对应这份完整文本
    if input_ids.ndim != 1 or input_ids.tolist() != encoded["input_ids"]:
        raise ValueError("input_ids 必须是这份完整对话的一维编码")

    # offsets 表示token 的位置 -》原始文本的字符位置
    offsets = encoded["offset_mapping"]

    # 2. 默认全部不参与 loss
    labels = torch.full_like(input_ids, -100)

    # 3. 逐条处理消息，因此能覆盖多轮 assistant 回答
    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue

        # 找到当前 assistant 正文开始之前的全部文本
        prefix = tokenizer.apply_chat_template(
            messages[:i],
            tokenize=False,
            add_generation_prompt=True,
        )

        start = len(prefix)
        end = start + len(message["content"])

        # 防止模板变化导致正文位置判断错误
        assert text.startswith(prefix)
        assert text[start:end] == message["content"]

        # 4. 找到完全落在正文范围内的 token
        for token_index, (s, e) in enumerate(offsets):
            if e > s and start <= s and e <= end:
                labels[token_index] = input_ids[token_index]

    return labels
    
if __name__ == "__main__":
    
    messages = [
        {"role": "system", "content": "你是一个助手。"},
        {"role": "user", "content": "1加1等于几？"},
        {"role": "assistant", "content": "2。"},
    ]

    text = format_messages(messages)
    print(text)
    
    encoded = tokenizer(
    text,
    add_special_tokens=False,
    return_tensors="pt",
)

    input_ids = encoded["input_ids"][0]

    print("token IDs：", input_ids)
    print("输入形状：", input_ids.shape)
    print("还原文本：", tokenizer.decode(input_ids.tolist()))
    
    labels = build_labels(input_ids, messages)

    print("labels：", labels)
    print("标签形状：", labels.shape)

    assert labels.shape == input_ids.shape