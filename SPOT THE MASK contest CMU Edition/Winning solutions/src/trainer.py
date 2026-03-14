import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(self, model: nn.Module, device: torch.device,
                 lr: float = 1e-4, patience: int = 5):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10
        )
        self.patience = patience
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0
        self.best_model_path = "models/best_model.pth"
        # CREATE DIRECTORY
        os.makedirs("models", exist_ok=True)

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train(train)
        total_loss = 0.0
        with torch.set_grad_enabled(train):
            for images, labels, _ in tqdm(loader, leave=False):
                images = images.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item() * len(images)
        return total_loss / len(loader.dataset)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int = 30) -> None:
        import os; os.makedirs("../models", exist_ok=True)
        for epoch in range(1, epochs + 1):
            train_loss = self._run_epoch(train_loader, train=True)
            val_loss = self._run_epoch(val_loader, train=False)
            self.scheduler.step()
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f" Saved best model (val_loss={val_loss:.4f})")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break