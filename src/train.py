"""
train.py
Training loop for A3T-GCN with early stopping and model checkpointing.
"""

import json
import torch
import numpy as np
from data_loader import load_config
from dataset import get_dataloaders
from model import build_model
from loss import DirectionalMSELoss


def train(cfg: dict):
    torch.manual_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    train_dl, val_dl, _ = get_dataloaders(cfg)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    criterion = DirectionalMSELoss(alpha=cfg["loss"]["alpha"])

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        # --- Train ---
        model.train()
        train_losses = []
        for x, adj, y in train_dl:
            x, adj, y = x.to(device), adj.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x, adj).squeeze(-1)   # (B, N)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validate ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, adj, y in val_dl:
                x, adj, y = x.to(device), adj.to(device), y.to(device)
                pred = model(x, adj).squeeze(-1)
                val_losses.append(criterion(pred, y).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # --- Early stopping & checkpointing ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), cfg["paths"]["model_checkpoint"])
        else:
            patience_counter += 1
            if patience_counter >= cfg["training"]["early_stopping_patience"]:
                print(f"[INFO] Early stopping at epoch {epoch}.")
                break

    # Save training history
    with open(cfg["paths"]["metrics"], "w") as f:
        json.dump(history, f)

    print(f"[INFO] Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    cfg = load_config()
    train(cfg)
