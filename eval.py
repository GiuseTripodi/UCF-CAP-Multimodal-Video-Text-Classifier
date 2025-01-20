import argparse
import sys
import os

from parse_config import ConfigParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from data_loader.ucf_cap_dataset import UCF101Dataset
from model.video_transformer import SpaceTimeTransformer
import logging
from logger import setup_logging

# Configure and create a logger
logger = logging.getLogger('eval')


# Define a function to load the trained model
def load_model(config, model_path):
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
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total Trainable Parameters: {total_params}")
    logger.info(f"Total Trainable Parameters: {total_params}")
    model.eval()
    return model


# Define a function to extract embeddings and predictions
def extract_predictions(model, dataloader, device):
    embeddings = []
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Get the embeddings and predictions
            features = model.forward_features(inputs)  # Extract features
            predictions = model(inputs)  # Get predictions
            pred_labels.append(torch.argmax(predictions, dim=1).cpu().numpy())
            embeddings.append(features.cpu().numpy())
            true_labels.append(labels.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    pred_labels = np.concatenate(pred_labels, axis=0)
    return embeddings, true_labels, pred_labels


# Define a function to compute classification metrics
def compute_metrics(true_labels, pred_labels):
    print("\n[INFO] Classification Metrics")
    accuracy = np.mean(true_labels == pred_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    report = classification_report(true_labels, pred_labels, zero_division=0)
    print("Classification Report:")
    print(report)

    conf_matrix = confusion_matrix(true_labels, pred_labels)
    print("Confusion Matrix:")
    print(conf_matrix)


# Main function
def eval_spacetime(config: ConfigParser, model_name):
    # Data transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset and dataloader
    test_path = config.test_path
    logger.info(f'Loading dataset from {test_path}')
    test_dataset = UCF101Dataset(test_path, transform=transform)
    test_dataloader = DataLoader(test_dataset, config.batch_size, config.shuffle)
    print('Test dataset:', len(test_dataset), 'samples')

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = os.path.join(config.save_dir, f'{model_name}')
    model = load_model(config, model_path).to(device)

    # Extract embeddings and predictions
    print('[INFO] Model Loaded')
    embeddings, true_labels, pred_labels = extract_predictions(model, test_dataloader, device)

    # Compute and display metrics
    compute_metrics(true_labels, pred_labels)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to train model")
    parser.add_argument('--config', default=None, help='Path to configuration file')
    parser.add_argument('--model_name', help='Path to CSV file with dataset information to eval the model')
    parser.add_argument('--name', default=None, help='Name of the experiment (used for saving the model)')
    parser.add_argument('--save_dir', default=None, help='Path to where get the saves file')
    args = parser.parse_args()

    # setup_logging(args.log)

    config = ConfigParser(args)
    logger = config.get_logger('Evaluation')
    logger.info("Evaluation started")
    eval_spacetime(config, args.model_name)