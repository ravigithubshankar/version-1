"""
MNIST Classification -- AlphaRed SDK demo.
Self-contained: data, model, training, SDK integration all in one file.

Usage:
    pip install torch torchvision requests psutil
    python train.py
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ── AlphaRed SDK ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alphared.run_session import RunSession

# ── Config ────────────────────────────────────────────────────────────────
EPOCHS         = 10
BATCH_SIZE     = 64
LR             = 0.001
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "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")
ALPHARED_HOST  = os.environ.get("ALPHARED_HOST", "http://localhost:8000")

# Agent subprocess లో infinite loop prevent చేయడానికి
ENABLE_AGENT   = not bool(os.environ.get("ALPHARED_AGENT_RUN"))


# ── Model ─────────────────────────────────────────────────────────────────
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


# ── Data ──────────────────────────────────────────────────────────────────
def get_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds     = datasets.MNIST("./data", train=True,  download=True, transform=transform)
    val_ds       = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


# ── Train one epoch ───────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)
    return total_loss / total, correct / total


# ── Validate ──────────────────────────────────────────────────────────────
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += images.size(0)
    return total_loss / total, correct / total


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    # ── AlphaRed: init session ────────────────────────────────────────────
    run = RunSession(
        project = "mnist-classification",
        name    = f"run-{int(time.time())}",
        host    = ALPHARED_HOST,
    )
    run.start()

    # ── AlphaRed: enable overnight agent (only in main run, not subprocess) ──
    if ENABLE_AGENT:
        run.enable_agent(
            openrouter_api_key  = OPENROUTER_KEY,
            train_file_path     = os.path.abspath(__file__),
            max_experiments     = 20,
            time_per_experiment = 180,
        )

    # ── Setup ─────────────────────────────────────────────────────────────
    train_loader, val_loader = get_loaders()
    model     = MNISTNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # ── AlphaRed: track model + datasets ─────────────────────────────────
    run.push(scope=locals(), model=model)

    print(f"Training on {DEVICE}  |  {EPOCHS} epochs  |  batch={BATCH_SIZE}  |  lr={LR}")
    print("-" * 60)

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss,   val_acc   = validate(model,   val_loader,   criterion, DEVICE)

        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        # ── AlphaRed: log metrics every epoch ────────────────────────────
        run.log({
            "epoch":          epoch,
            "train_loss":     train_loss,
            "train_accuracy": train_acc,
            "val_loss":       val_loss,
            "val_accuracy":   val_acc,
            "lr":             LR,
        })

    # ── AlphaRed: finish -- saves model, starts agent if enabled ─────────
    run.finish()
    print("\nTraining complete.")
    if ENABLE_AGENT:
        print("Agent running in background -- check alphared_agent/ for changes.md")
    else:
        print("Agent run complete (subprocess mode).")


if __name__ == "__main__":
    main()
