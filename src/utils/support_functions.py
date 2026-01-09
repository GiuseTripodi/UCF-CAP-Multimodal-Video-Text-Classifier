import numpy as np
import torch

from data_loader.ucf_cap_loader import UCF101Dataset
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst,
    ScaleIntensity, Resize, RandRotate90, RandFlip, RandZoom
)
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader
from collections import Counter

from src.utils.parse_config import ConfigParser

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


def load_dataset(config:ConfigParser):
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

    dataset_train = UCF101Dataset(csv_train_file_path, sampling_method='interpolate', num_frames=32, transform=train_transforms)
    dataset_val = UCF101Dataset(csv_val_file_path, sampling_method='interpolate', num_frames=32, transform=val_transforms)


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

