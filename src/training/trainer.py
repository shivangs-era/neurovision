"""
Training engine for Deep Learning models with validation, metric tracking, and checkpoints.
"""

import os
import time
from typing import Dict, Any, Optional, Callable, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from .metrics import compute_classification_metrics


class ModelTrainer:
    """
    Supervised trainer for classification neural networks.
    Features lr scheduling, progress tracking, validation metrics, and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        learning_rate: float = 1e-3,
        optimizer_type: str = "adam",
        weight_decay: float = 1e-4,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

        if optimizer_type.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer_type.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer_type.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "epoch_times": [],
        }

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = total_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, Any]]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for images, labels in val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

        val_loss = total_loss / max(len(all_targets), 1)
        metrics = compute_classification_metrics(np.array(all_targets), np.array(all_preds))
        return val_loss, metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 5,
        save_path: Optional[str] = None,
        callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Executes complete training loop across specified epochs.
        """
        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_metrics = self.evaluate(val_loader)
            val_acc = val_metrics["accuracy"]
            elapsed = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["epoch_times"].append(elapsed)

            epoch_stats = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_metrics["macro_f1"],
                "elapsed": elapsed,
            }

            if callback:
                callback(epoch, epochs, epoch_stats)

            # Checkpoint best model
            if val_acc > best_val_acc and save_path:
                best_val_acc = val_acc
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_acc": val_acc,
                        "history": self.history,
                    },
                    save_path,
                )

        return {
            "best_val_acc": best_val_acc,
            "final_metrics": val_metrics,
            "history": self.history,
        }


class VAETrainer:
    """
    Unsupervised generative trainer for the Variational Autoencoder.
    """

    def __init__(
        self,
        vae: nn.Module,
        device: Optional[torch.device] = None,
        learning_rate: float = 1e-3,
        beta: float = 1.0,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = vae.to(self.device)
        self.optimizer = optim.Adam(self.vae.parameters(), lr=learning_rate)
        self.beta = beta
        self.history: Dict[str, List[float]] = {
            "total_loss": [],
            "recon_loss": [],
            "kld": [],
        }

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.vae.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kld = 0.0
        total_samples = 0

        for images, _ in loader:
            images = images.to(self.device)
            self.optimizer.zero_grad()

            recon, mu, logvar = self.vae(images)
            loss_dict = self.vae.loss_function(recon, images, mu, logvar, beta=self.beta)
            loss_dict["loss"].backward()
            self.optimizer.step()

            batch_size = images.size(0)
            total_loss += loss_dict["loss"].item()
            total_recon += loss_dict["recon_loss"].item()
            total_kld += loss_dict["kld"].item()
            total_samples += batch_size

        return {
            "total_loss": total_loss / max(total_samples, 1),
            "recon_loss": total_recon / max(total_samples, 1),
            "kld": total_kld / max(total_samples, 1),
        }

    def fit(
        self,
        loader: DataLoader,
        epochs: int = 5,
        save_path: Optional[str] = None,
        callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
    ) -> Dict[str, Any]:
        for epoch in range(1, epochs + 1):
            stats = self.train_epoch(loader)
            self.history["total_loss"].append(stats["total_loss"])
            self.history["recon_loss"].append(stats["recon_loss"])
            self.history["kld"].append(stats["kld"])

            if callback:
                callback(epoch, epochs, stats)

            if save_path:
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                torch.save(self.vae.state_dict(), save_path)

        return {"history": self.history}
