import sys
import os
import collections

from sklearn.utils import compute_class_weight

from parse_config import ConfigParser
from trainer.trainer_video import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))
import argparse
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from model.video_transformer import *
from data_loader.ucf_cap_dataset import *
import logging
from logger import setup_logging
import model.metric as module_metric
from sklearn.metrics import accuracy_score

# Define a simple function to compute accuracy
def compute_metrics(p):
    preds, labels = p
    preds = torch.argmax(torch.tensor(preds), dim=1)  # Get class predictions
    accuracy = accuracy_score(labels, preds)
    return {"accuracy": accuracy}

def training(config: ConfigParser):
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset
    csv_train_file_path, csv_val_file_path = config.train_path
    logger.info(f'Loading dataset from {csv_train_file_path}')
    dataset_train = UCF101Dataset(csv_train_file_path, transform=transform)
    train_dataloader = DataLoader(dataset_train, config.batch_size, config.shuffle)
    dataset_val = UCF101Dataset(csv_val_file_path, transform=transform)
    val_dataloader = DataLoader(dataset_val, config.batch_size, config.shuffle)

    # Initialize model
    model = SpaceTimeTransformer(
        img_size=config.img_size,
        num_frames=config.num_frames,
        in_chans=config.in_chans,
        num_classes=config.num_classes,
        depth=config.depth,
        num_heads=config.num_heads,
        embed_dim=768,
        attention_style='frozen-in-time'
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total Trainable Parameters: {total_params}")
    logger.info(f"Total Trainable Parameters: {total_params}")

    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device model: {device}")
    model = model.to(device)

    # Define loss and optimizer and test class labels
    # Get class labels from the dataset
    train_labels = [label for _, label, *_ in dataset_train]  # Extract labels

    # Calculate class weights (inversely proportional to class frequencies)
    unique_labels = np.unique(train_labels)
    class_weights = compute_class_weight(class_weight='balanced', classes=unique_labels, y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    #class_weights = class_weights / class_weights.max()  # Normalize if necessary
    print(class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    logger.info(model)

    # Initialize Trainer class and start training
    trainer = Trainer(model, train_dataloader, val_dataloader, criterion, optimizer, device, config)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to train model")
    parser.add_argument('--name', default=None, help='Name of the experiment (used for saving the model)')
    parser.add_argument('--config', default=None, help='Path to configuration file')
    parser.add_argument('--save_dir', default=None, help='Path to where get the saves file')
    args = parser.parse_args()

    config = ConfigParser(args)
    logger = config.get_logger('Train')
    logger.info("Training started")
    training(config)