import os
import logging
from datetime import date
from os.path import join
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from src.utils.parse_config import ConfigParser


logger = logging.getLogger('train')


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, config: ConfigParser):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.exper_name = config.exper_name
        self.save_path = config.save_dir
        self.num_epochs = config.num_epochs
        self.config = config

        # Track training metrics
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_precisions = []
        self.val_recalls = []
        self.val_f1s = []

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for inputs, caption, labels, *other_info in tqdm(self.train_loader, desc="Training Epoch"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            if not isinstance(outputs, torch.Tensor):
                outputs = outputs.logits
            loss = self.criterion(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, caption, labels, *other_info in tqdm(self.val_loader, desc="Validation"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                if not isinstance(outputs, torch.Tensor):
                    outputs = outputs.logits

                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = running_loss / len(self.val_loader)

        # Compute metrics
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        return avg_loss, acc, prec, rec, f1

    def save_model(self):
        if self.config.modality == 1:
            path = join(self.save_path, f'space_time_{self.exper_name}_{date.today().strftime("%d-%m-%y")}')
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path)
        elif self.config.modality == 0:
            torch.save(self.model.state_dict(),
                       join(self.save_path, f'space_time_{self.exper_name}_{date.today().strftime("%d-%m-%y")}'))
            logger.info(f"Model saved to {self.save_path}")
        logger.info(f"Model {self.exper_name}_{date.today().strftime('%d-%m-%y')} saved to {self.save_path}")

    def plot_metrics(self):
        """Plot loss and accuracy curves."""
        plot_dir = join(self.save_path, "train_plots")
        os.makedirs(plot_dir, exist_ok=True)

        # Loss plot
        plt.figure(figsize=(8, 5))
        plt.plot(self.train_losses, label="Train Loss", marker='o')
        plt.plot(self.val_losses, label="Validation Loss", marker='s')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss Curve")
        plt.legend()
        plt.grid(True)
        loss_plot_path = join(plot_dir, f"loss_curve_{self.exper_name}_{date.today().strftime('%d-%m-%y')}.png")
        plt.savefig(loss_plot_path)
        plt.close()

        # Accuracy plot
        plt.figure(figsize=(8, 5))
        plt.plot(self.val_accuracies, label="Validation Accuracy", marker='o', color='green')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy Curve")
        plt.legend()
        plt.grid(True)
        acc_plot_path = join(plot_dir, f"accuracy_curve_{self.exper_name}_{date.today().strftime('%d-%m-%y')}.png")
        plt.savefig(acc_plot_path)
        plt.close()

        logger.info(f"Loss curve saved to {loss_plot_path}")
        logger.info(f"Accuracy curve saved to {acc_plot_path}")

    def train(self):
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {train_loss:.4f}")

            val_loss, acc, prec, rec, f1 = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(acc)
            self.val_precisions.append(prec)
            self.val_recalls.append(rec)
            self.val_f1s.append(f1)

            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}, "
                        f"Val Loss: {val_loss:.4f}, Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

        self.save_model()
        self.plot_metrics()
