import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, List


class MultimodalTrainer:
    """
    A trainer class for multimodal video-text models with training, validation,
    and checkpoint management capabilities.
    """

    def __init__(
            self,
            model: nn.Module,
            train_loader,
            val_loader,
            criterion: nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
            device: str = 'cuda',
            num_epochs: int = 100,
            save_dir: str = './',
            logger=None
    ):
        """
        Initialize the trainer.

        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            device: Device to train on ('cuda' or 'cpu')
            num_epochs: Number of training epochs
            save_dir: Directory to save checkpoints and results
            logger: Logger instance (optional)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.logger = logger

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_accuracy = 0.0
        self.current_epoch = 0

        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0

        train_loop = tqdm(
            self.train_loader,
            desc=f"Training Epoch {self.current_epoch + 1}/{self.num_epochs}",
            leave=True
        )

        for videos, texts, labels in train_loop:
            videos = videos.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(videos, texts)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self) -> Tuple[float, float]:
        """
        Validate the model.

        Returns:
            Tuple of (average validation loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        val_loop = tqdm(
            self.val_loader,
            desc=f"Validation Epoch {self.current_epoch + 1}/{self.num_epochs}",
            leave=True
        )

        with torch.no_grad():
            for videos, texts, labels in val_loop:
                videos = videos.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(videos, texts)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            filename: Name of the checkpoint file
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_accuracy': self.best_accuracy
        }

        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)

        if is_best and self.logger:
            self.logger.info(f"Saved best model with accuracy: {self.best_accuracy:.4f}")

    def load_checkpoint(self, filepath: str):
        """
        Load model checkpoint.

        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_accuracies = checkpoint['val_accuracies']
        self.best_accuracy = checkpoint['best_accuracy']

        if self.logger:
            self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

    def train(self):
        """
        Main training loop.
        """
        if self.logger:
            self.logger.info("Starting training...")

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss, val_accuracy = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)

            # Log metrics
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

            if self.logger:
                self.logger.info(
                    f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, "
                    f"Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.4f}"
                )

            # Save best model
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                self.save_checkpoint(f'best_model_.pth', is_best=True)
                print(f"✓ New best model saved with accuracy: {self.best_accuracy:.4f}")

            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')

            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()

        # Save training history and plot results
        self.save_training_history()
        self.plot_training_results()

        if self.logger:
            self.logger.info(f"Training complete! Best accuracy: {self.best_accuracy:.4f}")

    def save_training_history(self):
        """
        Save training history to a numpy file.
        """
        filepath = os.path.join(self.save_dir, 'training_history.npz')
        np.savez(
            filepath,
            train_losses=self.train_losses,
            val_losses=self.val_losses,
            val_accuracies=self.val_accuracies
        )
        print(f"\n✓ Training history saved to {filepath}")

    def plot_training_results(self):
        """
        Plot and save training results.
        """
        plt.figure(figsize=(14, 5))

        # Loss curves
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', marker='o', markersize=3)
        plt.plot(self.val_losses, label='Val Loss', marker='s', markersize=3)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)

        # Accuracy curve
        plt.subplot(1, 2, 2)
        plt.plot(self.val_accuracies, label='Val Accuracy', marker='o', markersize=3, color='green')
        plt.axhline(y=self.best_accuracy, color='r', linestyle='--', label=f'Best: {self.best_accuracy:.4f}')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Validation Accuracy', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(self.save_dir, 'training_results.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Training plots saved to {filepath}")
        plt.close()

    def get_metrics(self) -> Dict[str, List[float]]:
        """
        Get training metrics.

        Returns:
            Dictionary containing training metrics
        """
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_accuracy': self.best_accuracy
        }
