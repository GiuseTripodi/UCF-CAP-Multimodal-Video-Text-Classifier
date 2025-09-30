import sys
import os
from datetime import date
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
import argparse
import logging
import numpy as np

from torch.optim.lr_scheduler import StepLR  # 🔧 Added import

from data_loader.ucf_cap_dataset import UCF101Dataset
from model.MLP_classifier import MLPClassifier
from utils.utilis_combination_text_video import (
    extract_text_embeddings,
    extract_videos_embedding,
    load_text_encoder,
    load_model_embeddings,
    project_text_video, load_pretrained_text_model, extract_text_embeddings_weight,
    load_pretrained_text_model_with_embeddings,
)
from parse_config import ConfigParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))


def train_model_MLP(classifier, train_loader, val_loader, criterion, optimizer, scheduler, device, config, logger):
    """
    Train the MLP model using both training and validation datasets.
    """
    classifier.train()

    num_epochs = config.num_epochs
    best_val_loss = float("inf")
    save_path = f"{config.save_dir}/{config.exper_name}_best_MLP.pth"

    for epoch in range(num_epochs):
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        logger.info(f"Epoch [{epoch+1}/{num_epochs}] Training...")

        for inputs, caption, labels, text_embeddings, video_embeddings, *_ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            text_embeddings = text_embeddings.to(device).squeeze(1)
            video_embeddings = video_embeddings.to(device).squeeze(1)

            with torch.no_grad():
                combined_embedding = torch.cat((text_embeddings, video_embeddings), dim=-1)
                combined_embedding = combined_embedding.mean(dim=1).float()

            outputs = classifier(combined_embedding)
            loss = criterion(outputs, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

        train_accuracy = 100 * train_correct / train_total
        logger.info(f"Train Loss: {train_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")

        val_loss, val_accuracy = validate_model_MLP(classifier, val_loader, criterion, device, config, logger)
        logger.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(classifier.state_dict(), save_path)
            logger.info(f"Best model saved at {save_path}")

        scheduler.step()
        logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']}")  # 🔧 log current LR


def validate_model_MLP(classifier, val_loader, criterion, device, config, logger):
    classifier.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, caption, labels, text_embeddings, video_embeddings, *_ in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            text_embeddings = text_embeddings.to(device).squeeze(1)
            video_embeddings = video_embeddings.to(device).squeeze(1)

            combined_embedding = torch.cat((text_embeddings, video_embeddings), dim=-1)
            combined_embedding = combined_embedding.mean(dim=1).float()

            outputs = classifier(combined_embedding)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = 100 * val_correct / val_total
    return val_loss / len(val_loader), val_accuracy


def run_training(config: ConfigParser, model_name, expt_name):
    logger = config.get_logger('TrainMLP')
    logger.info(f'Starting training: {model_name}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_csv = '/Users/user/PycharmProjects/frozen-in-time/data/UcfCap/Train_embeddings_all.pkl'
    val_csv = '/Users/user/PycharmProjects/frozen-in-time/data/UcfCap/Val_embeddings_all.pkl'
    logger.info(f'Loading training dataset from {train_csv}')
    logger.info(f'Loading validation dataset from {val_csv}')
    dataset_train = UCF101Dataset(train_csv, transform=transform, num_samples=1000)
    train_dataloader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=config.shuffle)
    dataset_val = UCF101Dataset(val_csv, transform=transform, num_samples=200)
    val_dataloader = DataLoader(dataset_val, batch_size=config.batch_size, shuffle=config.shuffle)

    input_dim = 1536
    hidden_dim = 256
    num_classes = 12
    classifier = MLPClassifier(input_dim, hidden_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=config.learning_rate, weight_decay=1e-4)

    #StepLR scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Reduce LR by 10x every 10 epochs
    train_model_MLP(classifier, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device, config, logger)

    final_model_path = f"{config.save_dir}/{expt_name}_{date.today().strftime('%d-%m-%y')}_final_MLP.pth"
    torch.save(classifier.state_dict(), final_model_path)
    logger.info(f'Final model saved at {final_model_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to train MLP model with validation")
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--model_name', required=True, help='Name of the experiment')
    parser.add_argument('--save_dir', default=None, help='Path to where get the saves file')
    parser.add_argument('--name', default=None, help='Name of the experiment (used for saving the model)')
    args = parser.parse_args()

    config = ConfigParser(args)
    model_name = args.model_name
    expt_name = args.name

    run_training(config, model_name, expt_name)
