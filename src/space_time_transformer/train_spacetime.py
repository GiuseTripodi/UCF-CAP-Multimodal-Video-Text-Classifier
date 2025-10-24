import os
import sys
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
from src.utils.parse_config import ConfigParser



def load_model(config: ConfigParser, num_classes):
    model_name = "facebook/timesformer-base-finetuned-k400"
    model = TimesformerForVideoClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
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

    # Data transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    csv_train_file_path, csv_val_file_path = config.train_path
    logger.info(f'Loading dataset from {csv_train_file_path}')
    dataset_train = UCF101Dataset(csv_train_file_path)
    train_dataloader = DataLoader(dataset_train, config.batch_size, config.shuffle)
    dataset_val = UCF101Dataset(csv_val_file_path)
    val_dataloader = DataLoader(dataset_val, config.batch_size, config.shuffle)

    # Define loss and optimizer and test class labels
    # Get class labels from the dataset
    train_labels = [label for _, _, label, *_ in dataset_train]  # Extract labels

    # Calculate class weights (inversely proportional to class frequencies)
    unique_labels = np.unique(train_labels)
    class_weights = compute_class_weight(class_weight='balanced', classes=unique_labels, y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(class_weights)

    # Load the model
    logger.info(f'Loading model')
    model, processor = load_model(config, num_classes=len(class_weights))
    model = model.to(device)



    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    trainer = Trainer(model, train_dataloader, val_dataloader, criterion, optimizer, device, config)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to train model")
    parser.add_argument('--name', default='Test', help='Name of the experiment (used for saving the model)')
    parser.add_argument('--config', default='/Users/user/PycharmProjects/frozen-in-time/configs/ucf-cap.json', help='Path to configuration file')
    parser.add_argument('--save_dir', default='/Users/user/PycharmProjects/frozen-in-time/data', help='Path to where get the saves file')
    args = parser.parse_args()

    config = ConfigParser(args)
    training(config)