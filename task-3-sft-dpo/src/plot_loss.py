"""读取 SFT 每轮 loss 日志，生成训练集与验证集曲线。"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

# 直接输出图片，在没有图形界面的环境中也能运行。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]


def load_history(history_path):
    """JSON 文件 -> 三个列表：epoch、train_loss、val_loss。"""
    with history_path.open(encoding="utf-8") as file:
        payload = json.load(file)
    metrics = payload.get("metrics") if isinstance(payload, dict) else None
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("日志中的 metrics 必须是非空列表")

    epochs, train_losses, val_losses = [], [], []
    for row in metrics:
        if not isinstance(row, dict) or not {"epoch", "train_loss", "val_loss"} <= row.keys():
            raise ValueError("每条记录必须包含 epoch、train_loss 和 val_loss")
        epoch = row["epoch"]
        if type(epoch) is not int or epoch < 1 or (epochs and epoch <= epochs[-1]):
            raise ValueError("epoch 必须是从 1 开始、严格递增的整数")
        train_loss = float(row["train_loss"])
        val_loss = float(row["val_loss"])
        if not all(math.isfinite(loss) for loss in (train_loss, val_loss)):
            raise ValueError(f"epoch {epoch} 的 loss 出现 NaN 或 Inf，请检查训练")
        epochs.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    return epochs, train_losses, val_losses


def plot_history(history_path, output_path):
    epochs, train_losses, val_losses = load_history(history_path)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.plot(epochs, train_losses, marker="o", label="Train (during epoch)")
    ax.plot(epochs, val_losses, marker="o", label="Validation (after epoch)")

    # 用验证集选择最佳轮次，保留原始曲线，不做平滑。
    best_index = min(range(len(val_losses)), key=lambda index: val_losses[index])
    ax.scatter(
        epochs[best_index], val_losses[best_index],
        s=100, facecolors="none", edgecolors="black", zorder=3,
        label=f"Best val: epoch {epochs[best_index]}, {val_losses[best_index]:.4f}",
    )
    ax.set(
        title="SFT training and validation loss",
        xlabel="Epoch",
        ylabel="Token-averaged cross-entropy loss",
    )
    if len(epochs) <= 20:
        ax.set_xticks(epochs)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.25)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return epochs[best_index], val_losses[best_index]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "history", nargs="?", type=Path,
        help="历史 JSON 路径；省略时读取 runs/sft 中最近更新的日志",
    )
    parser.add_argument("--output", type=Path, help="图片路径；默认与 JSON 同名，后缀为 .png")
    args = parser.parse_args()

    history_path = args.history
    if history_path is None:
        candidates = list((ROOT / "runs" / "sft").glob("history_*.json"))
        if not candidates:
            parser.error("没有找到训练日志，请先训练至少一个完整 epoch，或传入 JSON 路径")
        history_path = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    output_path = args.output or history_path.with_suffix(".png")
    try:
        best_epoch, best_loss = plot_history(history_path, output_path)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(f"History: {history_path}")
    print(f"Plot: {output_path}")
    print(f"Best validation loss: {best_loss:.4f} (epoch {best_epoch})")


if __name__ == "__main__":
    main()
