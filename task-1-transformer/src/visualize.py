"""Generate attention heatmaps for positive, negative and long reviews.

Run from the task directory with::

    python -m src.visualize
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .model import load_for_eval


ROOT = Path(__file__).resolve().parents[1]


def predict(model, tokenize_fn, text):
    ids = tokenize_fn(text)
    with torch.no_grad():
        logits = model(ids.unsqueeze(0))
    return int(logits.argmax(dim=-1).item())


def choose_correct_sample(dataframe, model, tokenize_fn, label=None, longest=False):
    candidates = dataframe.copy()
    if label is not None:
        candidates = candidates[candidates["label"] == label]

    lengths = candidates["text"].astype(str).str.len()
    if longest:
        candidates = candidates.assign(text_length=lengths).sort_values(
            "text_length", ascending=False
        )
    else:
        # Medium-length examples keep the axes readable while still containing
        # enough context to show non-local attention patterns.
        candidates = candidates[(lengths >= 15) & (lengths <= 40)]

    for _, row in candidates.iterrows():
        text = str(row["text"])
        true_label = int(row["label"])
        if predict(model, tokenize_fn, text) == true_label:
            return text, true_label

    raise RuntimeError("找不到符合条件且预测正确的可视化样本")


def plot_attention(
    model,
    tokenize_fn,
    text,
    true_label,
    name,
    layer,
    head,
    max_plot_tokens,
):
    ids = tokenize_fn(text)
    with torch.no_grad():
        logits, all_attentions = model(
            ids.unsqueeze(0), return_attentions=True
        )

    layer_index = layer % len(all_attentions)
    weights = all_attentions[layer_index]
    if head < 0 or head >= weights.shape[1]:
        raise ValueError(f"head 必须在 0 到 {weights.shape[1] - 1} 之间")

    prediction = int(logits.argmax(dim=-1).item())
    token_count = min(len(text), model.max_len, max_plot_tokens)
    tokens = list(text[:token_count])
    matrix = weights[0, head, :token_count, :token_count].cpu().numpy()

    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Arial Unicode MS",
        "Heiti TC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    side = max(7.0, min(16.0, token_count * 0.25))
    figure, axis = plt.subplots(figsize=(side, side))
    image = axis.imshow(matrix, cmap="magma", vmin=0.0, aspect="auto")
    axis.set_xticks(np.arange(token_count), labels=tokens, rotation=90, fontsize=7)
    axis.set_yticks(np.arange(token_count), labels=tokens, fontsize=7)
    axis.set_xlabel("Key：被关注的字符")
    axis.set_ylabel("Query：发起查询的字符")
    axis.set_title(
        f"{name} | layer={layer_index}, head={head}, "
        f"label={true_label}, prediction={prediction}"
    )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()

    output_path = ROOT / "figures" / f"{name}.png"
    output_path.parent.mkdir(exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"saved: {output_path.relative_to(ROOT)}")
    print(f"text: {text[:model.max_len]}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--max-plot-tokens", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    model, tokenize_fn = load_for_eval(str(ROOT / "ckpt" / "best.pt"))
    model.eval()
    dev = pd.read_parquet(ROOT / "data" / "validation.parquet")

    positive = choose_correct_sample(dev, model, tokenize_fn, label=1)
    negative = choose_correct_sample(dev, model, tokenize_fn, label=0)
    long_sample = choose_correct_sample(
        dev, model, tokenize_fn, longest=True
    )

    for name, (text, true_label) in (
        ("positive", positive),
        ("negative", negative),
        ("long", long_sample),
    ):
        plot_attention(
            model=model,
            tokenize_fn=tokenize_fn,
            text=text,
            true_label=true_label,
            name=name,
            layer=args.layer,
            head=args.head,
            max_plot_tokens=args.max_plot_tokens,
        )


if __name__ == "__main__":
    main()
