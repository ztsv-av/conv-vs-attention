import math
from typing import Dict

import matplotlib.pyplot as plt
import torch

from vars import DEVICE
from data import CornerSubsetDataset


def visualize_dataset_samples(dataset, num_samples: int, out_path: str):
    num_samples = min(num_samples, len(dataset))

    # layout
    num_cols = max(1, num_samples // 4)  # avoid 0 columns
    num_rows = math.ceil(num_samples / num_cols)

    indices = torch.randint(0, len(dataset), (num_samples,)).tolist()

    plt.figure(figsize=(3 * num_cols, 3 * num_rows))

    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        img_np = img.squeeze(0).numpy()

        ax = plt.subplot(num_rows, num_cols, i + 1)
        ax.imshow(img_np, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"label={label}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_losses_accs(histories: Dict[str, dict], out_path: str):
    plt.figure(figsize=(12, 5))

    # left subplot: losses
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    # right subplot: accuracies
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")

    for i, (exp_name, hist) in enumerate(histories.items()):
        color = f"C{i}"
        epochs = range(1, len(hist["train_loss"]) + 1)

        # losses: train solid, test dashed
        ax1.plot(epochs, hist["train_loss"], color=color, label=f"{exp_name} train")
        ax1.plot(epochs, hist["test_loss"], color=color, linestyle="--", label=f"{exp_name} test")

        # accuracies: train solid, test dashed
        ax2.plot(epochs, hist["train_acc"], color=color, label=f"{exp_name} train")
        ax2.plot(epochs, hist["test_acc"], color=color, linestyle="--", label=f"{exp_name} test")

    ax1.legend()
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved losses/accuracies plot to {out_path}")


def plot_runtimes(histories: Dict[str, dict], out_path: str):
    exp_names = list(histories.keys())
    train_times = [histories[n]["train_time"] for n in exp_names]
    infer_times = [histories[n]["inference_time"] for n in exp_names]

    x = range(len(exp_names))

    plt.figure(figsize=(10, 4))

    # training times
    ax1 = plt.subplot(1, 2, 1)
    ax1.bar(x, train_times)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(exp_names, rotation=45, ha="right")
    ax1.set_ylabel("Seconds")
    ax1.set_title("Training time per experiment")

    # inference times
    ax2 = plt.subplot(1, 2, 2)
    ax2.bar(x, infer_times)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(exp_names, rotation=45, ha="right")
    ax2.set_ylabel("Seconds")
    ax2.set_title("Single-sample inference time")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved runtime plot to {out_path}")


def plot_sample_predictions(
    models: Dict[str, torch.nn.Module],
    test_dataset: CornerSubsetDataset,
    out_path: str,
    device: str = DEVICE,
    num_samples: int = 8,
):
    indices = list(range(min(num_samples, len(test_dataset))))
    exp_names = list(models.keys())

    n_rows = len(exp_names)
    n_cols = len(indices)

    plt.figure(figsize=(2 * n_cols, 2.5 * n_rows))

    for row, exp_name in enumerate(exp_names):
        model = models[exp_name].to(device)
        model.eval()
        for col, idx in enumerate(indices):
            img, label = test_dataset[idx]
            inp = img.unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(inp)
                pred = logits.argmax(dim=1).item()

            ax = plt.subplot(n_rows, n_cols, row * n_cols + col + 1)
            ax.imshow(img.squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
            if col == 0:
                ax.set_ylabel(exp_name, fontsize=8)
            ax.set_title(f"T:{label} P:{pred}", fontsize=8)
            ax.axis("off")

    plt.suptitle("Sample predictions for all experiments", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved prediction plot to {out_path}")
