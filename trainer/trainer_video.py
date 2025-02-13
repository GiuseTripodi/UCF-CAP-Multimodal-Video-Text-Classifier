import sys
import os
from datetime import date
from os.path import join

import torch
import torch.optim as optim
import torch.nn as nn
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
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, config:ConfigParser):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.exper_name = config.exper_name
        self.save_path = config.save_dir
        self.num_epochs = config.epochs
        self.config = config

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for inputs, labels, *other_info in tqdm(self.train_loader, desc="Training Epoch"):
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
            for inputs, labels, *other_info in tqdm(self.val_loader, desc="Validation"):
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
            # Save model after all epochs are completed
            torch.save(self.model.state_dict(), join(self.save_path, f'space_time_{self.exper_name}_{date.today().strftime("%d-%m-%y")}'))
            logger.info(f"Model saved to {self.save_path}")
        logger.info(f"Model {self.exper_name}_{date.today().strftime('%d-%m-%y')} saved to {self.save_path}")



    def train(self):
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch()
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {train_loss:.4f}")

            val_loss = self.validate()
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Validation Loss: {val_loss:.4f}")

        # Save the model after training is complete
        self.save_model()
