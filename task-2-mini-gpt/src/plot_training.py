"""保存训练指标，并绘制每轮 loss / dev perplexity。"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 保存图片，不弹出窗口，也支持无桌面的训练服务器
import matplotlib.pyplot as plt


def plot_history(history, output_path):
    if not history:
        raise ValueError("没有可绘制的训练记录")

    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    dev_losses = [row["dev_loss"] for row in history]
    dev_ppls = [row["dev_ppl"] for row in history]
    best = min(range(len(history)), key=lambda i: dev_losses[i])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    axes[0].plot(epochs, train_losses, label="Train loss", marker=".")
    axes[0].plot(epochs, dev_losses, label="Dev loss", marker=".")
    axes[0].scatter(epochs[best], dev_losses[best], color="black", zorder=3,
                    label=f"Best dev: epoch {epochs[best]}")
    axes[0].set(title="Cross-entropy loss", ylabel="Loss per token")
    axes[0].legend()

    axes[1].plot(epochs, dev_ppls, color="tab:orange", marker=".", label="Dev PPL")
    axes[1].scatter(epochs[best], dev_ppls[best], color="black", zorder=3,
                    label=f"Best: {dev_ppls[best]:.2f} (epoch {epochs[best]})")
    axes[1].set(title="Dev perplexity (training validation windows)",
                ylabel="Perplexity (log scale)", yscale="log")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output_path, dpi=160)
    finally:
        plt.close(fig)


def save_training_history(history, config, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 先存数值，即使绘图失败，也能用下面的命令重新生成图片。
    (output_dir / "history.json").write_text(
        json.dumps({"config": config, "history": history},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_history(history, output_dir / "training_curves.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 history.json 重新绘制训练曲线")
    parser.add_argument("history_path", type=Path)
    args = parser.parse_args()
    data = json.loads(args.history_path.read_text(encoding="utf-8"))
    output_path = args.history_path.parent / "training_curves.png"
    plot_history(data["history"], output_path)
    print(f"曲线已保存：{output_path}")
