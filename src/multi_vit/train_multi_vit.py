import sys
import os
from datetime import date
import argparse
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parse_config import ConfigParser
from multimodal_vit import MultimodalSpaceTimeTransformer
from data_loader.lung_pet_ct import DICOMVolumeDataset


class MultimodalDataset(torch.utils.data.Dataset):
    """Wrapper to load video and text pairs"""

    def __init__(self, dicom_dataset, text_descriptions):
        self.dicom_dataset = dicom_dataset
        self.text_descriptions = text_descriptions
        assert len(dicom_dataset) == len(text_descriptions), "Dataset and text descriptions must have same length"

    def __len__(self):
        return len(self.dicom_dataset)

    def __getitem__(self, idx):
        video, label = self.dicom_dataset[idx]
        text = self.text_descriptions[idx]
        return video, text, label


def load_dataset(config):
    """Load DICOM videos and associated text descriptions"""
    csv_train_file_path, csv_val_file_path, csv_test_file_path = config.datasets_path(
        label=config.label_experiments, num_frames=config.num_frames
    )

    # Load video datasets
    from monai.transforms import (
        Compose, LoadImage, EnsureChannelFirst,
        ScaleIntensity, Resize, RandRotate90, RandFlip, RandZoom
    )

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

    dataset_train = DICOMVolumeDataset(csv_train_file_path, transform=train_transforms)
    dataset_val = DICOMVolumeDataset(csv_val_file_path, transform=val_transforms)

    # TODO: Load text descriptions from file or generate them
    # For now, using placeholder descriptions
    train_texts = [f"CT scan sample {i}" for i in range(len(dataset_train))]
    val_texts = [f"CT scan sample {i}" for i in range(len(dataset_val))]

    # Wrap with text
    dataset_train = MultimodalDataset(dataset_train, train_texts)
    dataset_val = MultimodalDataset(dataset_val, val_texts)

    train_loader = DataLoader(
        dataset_train, batch_size=config.batch_size, shuffle=True,
        num_workers=0, collate_fn=multimodal_collate_fn
    )

    val_loader = DataLoader(
        dataset_val, batch_size=config.batch_size, shuffle=False,
        num_workers=0, collate_fn=multimodal_collate_fn
    )

    return train_loader, val_loader


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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Training started: multimodal_{config.exper_name}_{date.today().strftime("%d-%m-%y")}')

    # Load datasets
    logger.info('Loading datasets...')
    train_loader, val_loader = load_dataset(config)

    # Get number of classes
    num_classes = len(train_loader.dataset.dicom_dataset.label_to_idx)
    print(f"Training with {num_classes} classes: {train_loader.dataset.dicom_dataset.label_to_idx}")

    # Calculate class weights for imbalance
    train_labels = [train_loader.dataset.dicom_dataset.data.iloc[i]['CancerType']
                    for i in range(len(train_loader.dataset.dicom_dataset))]
    label_counts = Counter(train_labels)

    print("\n=== Class Distribution ===")
    total_samples = len(train_labels)
    class_weights = []
    for i in range(num_classes):
        label_name = [k for k, v in train_loader.dataset.dicom_dataset.label_to_idx.items() if v == i][0]
        count = label_counts[label_name]
        weight = total_samples / (num_classes * count)
        class_weights.append(weight)
        print(f"  {label_name} ({i}): {count} samples ({100 * count / total_samples:.1f}%) - weight: {weight:.2f}")

    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Initialize model
    logger.info('Loading model...')
    model = MultimodalSpaceTimeTransformer(
        img_size=96,
        patch_size=16,
        in_chans=1,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_frames=8,
        text_model='distilbert-base-uncased',
        fusion_method='concat'
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # Training loop
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_accuracy = 0
    best_model_path = 'best_multimodal_model.pth'

    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")

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
    parser.add_argument('--config', default='config/lung_pet.json', help='Config file path')
    parser.add_argument('--save_dir', default='saves', help='Save directory')
    parser.add_argument('--label_experiments', default='ALL', help='Label type')
    parser.add_argument('--dataset_samples', default=100, help='Number of samples')
    args = parser.parse_args()

    config = ConfigParser(args)
    training(config)