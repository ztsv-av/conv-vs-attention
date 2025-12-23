import os
import time
from typing import Dict, Any

import torch
import torch.nn.functional as F

from config.vars import DEVICE, LR, WEIGHTS_DIR


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += imgs.size(0)

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)

            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += imgs.size(0)

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc


def run_experiment(
    exp_name: str,
    model,
    train_loader,
    test_loader,
    num_epochs: int,
    device: str = DEVICE,
    lr: float = LR,
) -> Dict[str, Any]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
        "train_time": None,
        "inference_time": None,
    }

    print(f"\n=== Experiment: {exp_name} ({num_epochs} epochs) ===")
    t_train_start = time.time()

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        print(
            f"[{exp_name}] Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f} | "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.3f}"
        )

    # training time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_train_end = time.time()
    train_time = t_train_end - t_train_start
    history["train_time"] = train_time
    print(f"[{exp_name}] Training time: {train_time:.3f} s")

    # inference time on a single sample (first sample from test loader)
    single_sample = None
    for imgs, labels in test_loader:
        single_sample = imgs[0:1].to(device)
        break

    if single_sample is not None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_inf_start = time.time()
        with torch.no_grad():
            _ = model(single_sample)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_inf_end = time.time()
        inf_time = t_inf_end - t_inf_start
        history["inference_time"] = inf_time
        print(f"[{exp_name}] Single-sample inference time: {inf_time:.6f} s")

    # save weights
    save_path = os.path.join(WEIGHTS_DIR, f"{exp_name}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Saved weights to {save_path}")

    return history
