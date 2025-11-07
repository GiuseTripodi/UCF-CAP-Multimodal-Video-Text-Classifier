import sys
import os
from datetime import date
import argparse
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from model.multi_vit import MultimodalSpaceTimeTransformer
from src.utils.parse_config import ConfigParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader.ucf_cap_loader import UCF101Dataset
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst,
    ScaleIntensity, Resize, RandRotate90, RandFlip, RandZoom
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def load_dataset(config):
    """Load DICOM videos and associated text descriptions"""
    csv_train_file_path, csv_val_file_path = config.train_path

    train_transforms = Compose([
        LoadImage(image_only=True, reader="ITKReader"),
        EnsureChannelFirst(),
        ScaleIntensity(minv=0.0, maxv=1.0),
        Resize((96, 96, 96)),
        RandRotate90(prob=0.5, spatial_axes=(1, 2)),
        RandFlip(prob=0.5, spatial_axis=0),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
    ])

    val_transforms = Compose([
        LoadImage(image_only=True, reader="ITKReader"),
        EnsureChannelFirst(),
        ScaleIntensity(minv=0.0, maxv=1.0),
        Resize((96, 96, 96)),
    ])

    dataset_train = UCF101Dataset(csv_train_file_path, sampling_method='uniform')
    dataset_val = UCF101Dataset(csv_val_file_path, sampling_method='uniform')


    train_loader = DataLoader(
        dataset_train, batch_size=config.batch_size, shuffle=True,
        num_workers=0, collate_fn=multimodal_collate_fn
    )

    val_loader = DataLoader(
        dataset_val, batch_size=config.batch_size, shuffle=False,
        num_workers=0, collate_fn=multimodal_collate_fn
    )

    # Define loss and optimizer and test class labels
    # Get class labels from the dataset
    train_labels = [label for _, _, label, *_ in dataset_train]  # Extract labels

    # Calculate class weights (inversely proportional to class frequencies)
    unique_labels = np.unique(train_labels)
    class_weights = compute_class_weight(class_weight='balanced', classes=unique_labels, y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(class_weights)


    return train_loader, val_loader, class_weights


def multimodal_collate_fn(batch):
    """Custom collate function for multimodal data"""
    videos = []
    texts = []
    labels = []

    for video, text, label in batch:
        videos.append(video)
        texts.append(text)
        labels.append(label)

    videos = torch.stack(videos)
    labels = torch.tensor(labels, dtype=torch.long)

    return videos, texts, labels


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    train_loop = tqdm(train_loader, desc="Training", leave=True)
    for videos, texts, labels in train_loop:
        videos = videos.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(videos, texts)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        train_loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    val_loop = tqdm(val_loader, desc="Validation", leave=True)
    with torch.no_grad():
        for videos, texts, labels in val_loop:
            videos = videos.to(device)
            labels = labels.to(device)

            logits = model(videos, texts)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def training(config: ConfigParser):
    logger = config.get_logger('Train')
    logger.info(f'Training started: multimodal_{config.exper_name}_{date.today().strftime("%d-%m-%y")}')

    # Load datasets
    logger.info('Loading datasets...')
    train_loader, val_loader, class_weights = load_dataset(config)

    # Initialize model
    logger.info('Loading model...')
    model = MultimodalSpaceTimeTransformer(
        num_classes=len(class_weights),
        text_model='distilbert-base-uncased',
        fusion_method='concat'
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    # Training loop
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_accuracy = 0
    best_model_path = 'best_multimodal_model.pth'

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        # Validate
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with accuracy: {best_accuracy:.4f}")

        scheduler.step()

    # Save training history
    np.savez('multimodal_training_history.npz',
             train_losses=train_losses,
             val_losses=val_losses,
             val_accuracies=val_accuracies)

    # Plot results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')

    plt.tight_layout()
    plt.savefig('multimodal_training_results.png')
    print("\nTraining complete! Results saved to multimodal_training_results.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multimodal training script")
    parser.add_argument('--name', default='Test', help='Experiment name')
    parser.add_argument('--config', default='/Users/user/PycharmProjects/frozen-in-time/configs/ucf-cap.json', help='Config file path')
    parser.add_argument('--save_dir', default='/Users/user/PycharmProjects/frozen-in-time/data', help='Save directory')
    parser.add_argument('--label_experiments', default='ALL', help='Label type')
    parser.add_argument('--dataset_samples', default=100, help='Number of samples')
    args = parser.parse_args()

    config = ConfigParser(args)
    training(config)