import os
import sys

from src.utils.parse_config import ConfigParser
from src.utils.support_functions import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import date
import argparse
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from data_loader.ucf_cap_loader import UCF101Dataset
from model.video_transformer import *
from sklearn.metrics import accuracy_score
from sklearn.utils import compute_class_weight
from transformers import AutoImageProcessor, TimesformerForVideoClassification, TimesformerConfig
from src.trainers.trainer_video import Trainer
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst,
    ScaleIntensity, Resize, RandRotate90, RandFlip, RandZoom
)


def load_model(num_classes):
    model_name = "facebook/timesformer-base-finetuned-k400"
    model = TimesformerForVideoClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            output_hidden_states=True

    )

    # Unfreeze the last few layers of the transformer backbone
    for name, param in model.named_parameters():
        if 'encoder.layer.10' in name or 'encoder.layer.11' in name:  # Adjust this based on your model depth
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Unfreeze the classification head
    for param in model.classifier.parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())  # Total parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # Trainable parameters
    frozen_params = total_params - trainable_params  # Frozen parameters

    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Frozen Parameters: {frozen_params:,}")

    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, processor

# Define a simple function to compute accuracy
def compute_metrics(p):
    preds, labels = p
    preds = torch.argmax(torch.tensor(preds), dim=1)  # Get class predictions
    accuracy = accuracy_score(labels, preds)
    return {"accuracy": accuracy}

def training(config: ConfigParser):
    logger = config.get_logger('Train')
    logger.info(f'Training started for model: space_time_{config.model_name}_{date.today().strftime("%d-%m-%y")}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device model: {device}")

    # Load datasets
    logger.info('Loading datasets...')
    train_loader, val_loader, class_weights = load_dataset(config)


    # Load the model
    logger.info(f'Loading model')
    model, processor = load_model(num_classes=len(class_weights))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device, config)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to train model")
    parser.add_argument('--name', default='Test', help='Name of the experiment (used for saving the model)')
    parser.add_argument('--config', default='/Users/user/PycharmProjects/frozen-in-time/configs/ucf-cap.json', help='Path to configuration file')
    parser.add_argument('--save_dir', default='/Users/user/PycharmProjects/frozen-in-time/data', help='Path to where get the saves file')
    args = parser.parse_args()

    config = ConfigParser(args)
    training(config)