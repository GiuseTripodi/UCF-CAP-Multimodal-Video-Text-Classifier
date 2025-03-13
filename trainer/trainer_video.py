import sys
import os
from datetime import date
from os.path import join

import torch
import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse
from data_loader.ucf_cap_dataset import *
from model.video_transformer import *
import logging
from logger import setup_logging
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from parse_config import ConfigParser

# Configure and create a logger
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

        # Store loss values for plotting
        self.train_losses = []
        self.val_losses = []

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
        with torch.no_grad():
            for inputs, caption, labels, *other_info in tqdm(self.val_loader, desc="Validation"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                if not isinstance(outputs, torch.Tensor):
                    outputs = outputs.logits
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

        avg_loss = running_loss / len(self.val_loader)
        return avg_loss

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

    def plot_losses(self):
        """ Plot training and validation loss after training """
        plt.figure(figsize=(8, 5))
        plt.plot(self.train_losses, label="Training Loss", marker='o')
        plt.plot(self.val_losses, label="Validation Loss", marker='s')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss Curve")
        plt.legend()
        plt.grid(True)

        # Save the plot
        plot_path = join(self.save_path, "train_plots",  f"loss_curve_{self.exper_name}_{date.today().strftime('%d-%m-%y')}.png")
        plt.savefig(plot_path)
        plt.show()
        logger.info(f"Loss curve saved to {plot_path}")

    def train(self):
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)  # Store training loss
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {train_loss:.4f}")

            val_loss = self.validate()
            self.val_losses.append(val_loss)  # Store validation loss
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}, Validation Loss: {val_loss:.4f}")

        # Save the model after training is complete
        self.save_model()

        # Plot the loss curves
        self.plot_losses()