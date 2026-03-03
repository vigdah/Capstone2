"""
train_mlp.py — Train a PyTorch MLP classifier for malware detection.

Architecture: Input → [128, 64, 32] hidden layers (ReLU + Dropout) → 2-class softmax

Usage:
  python train_mlp.py --splits_dir ../data/splits --model_dir ../models
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, roc_auc_score


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(128, 64, 32), dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hdim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())
    return np.array(all_probs), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser(description="Train MLP malware classifier")
    parser.add_argument("--splits_dir", default="../data/splits")
    parser.add_argument("--model_dir", default="../models")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    print("Loading splits...")
    X_train = np.load(os.path.join(args.splits_dir, "X_train.npy"))
    X_test = np.load(os.path.join(args.splits_dir, "X_test.npy"))
    y_train = np.load(os.path.join(args.splits_dir, "y_train.npy"))
    y_test = np.load(os.path.join(args.splits_dir, "y_test.npy"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # DataLoaders
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    input_dim = X_train.shape[1]
    model = MLP(input_dim=input_dim).to(device)
    print(f"MLP input_dim={input_dim}, parameters={sum(p.numel() for p in model.parameters()):,}")

    # Weighted loss for class imbalance
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    weight = torch.tensor([1.0, neg / (pos + 1e-8)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_auc = 0.0
    best_model_state = None

    print("\nTraining MLP...")
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        test_probs, test_labels = evaluate(model, test_loader, device)
        auc = roc_auc_score(test_labels, test_probs)
        scheduler.step(1 - auc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {loss:.4f} | AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Restore best model
    model.load_state_dict(best_model_state)
    print(f"\nBest AUC: {best_auc:.4f}")

    # Final evaluation
    test_probs, test_labels = evaluate(model, test_loader, device)
    y_pred = (test_probs >= 0.5).astype(int)
    print("\n=== MLP Evaluation ===")
    print(classification_report(test_labels, y_pred, target_names=["Benign", "Malicious"]))
    print(f"AUC-ROC: {roc_auc_score(test_labels, test_probs):.4f}")

    # Save PyTorch model
    torch.save(model.state_dict(), os.path.join(args.model_dir, "mlp_weights.pt"))

    # Save model config for ONNX export
    import json
    config = {"input_dim": input_dim, "hidden_dims": [128, 64, 32], "dropout": 0.3}
    with open(os.path.join(args.model_dir, "mlp_config.json"), "w") as f:
        json.dump(config, f)

    print(f"Model saved to {args.model_dir}/mlp_weights.pt")

    # Save MLP probability outputs for stacking
    train_loader_noshuffle = DataLoader(train_ds, batch_size=args.batch_size)
    train_probs, _ = evaluate(model, train_loader_noshuffle, device)
    np.save(os.path.join(args.splits_dir, "mlp_train_probs.npy"), train_probs)
    np.save(os.path.join(args.splits_dir, "mlp_test_probs.npy"), test_probs)
    print("Saved MLP probability outputs for stacking.")


if __name__ == "__main__":
    main()
