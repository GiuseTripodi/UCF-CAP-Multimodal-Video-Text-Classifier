import sys
import os
from datetime import date

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import torch

from data_loader.ucf_cap_dataset import UCF101Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from parse_config import ConfigParser
import argparse
import logging
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from model.video_transformer import SpaceTimeTransformer
import pandas as pd
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, TimesformerForVideoClassification


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

# Define a function to extract embeddings and predictions
def extract_embeddings_and_predictions(model, processor, dataloader, device):
    embeddings = []
    labels = []
    predictions = []
    with torch.no_grad():
        for inputs, label, *other_info in dataloader:
            inputs = inputs.to(device)
            if processor:
                inputs = inputs.squeeze(0)  # Removes the batch dimension
                inputs = processor(list(inputs), return_tensors="pt")

            # Get the embeddings and predicted classes
            outputs = model(inputs)
            logits = outputs.logits
            predicted_labels = torch.argmax(logits, dim=1)  # Get the predicted class

            #embeddings.append(embedding.cpu().numpy())
            labels.append(label.numpy())
            predictions.append(predicted_labels.cpu().numpy())

    #embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    return embeddings, labels, predictions


def load_model(config: ConfigParser, model_name, logger):
    if config.modality == 0:
        # Use model fine_tuned
        model_path = os.path.join(config.save_dir, f'{model_name}')
        logger.info(f"[INFO] Loaded fine_tuned model from: {model_path}")
        model = TimesformerForVideoClassification.from_pretrained(model_path)
        # processor = AutoImageProcessor.from_pretrained(model_path)
        processor = None
    elif config.modality == 1:
        logger.info(f"[INFO] Loaded pre-trained model: {config.model_name}")
        #Use pre-trained model
        processor = AutoImageProcessor.from_pretrained(config.model_name)
        processor = None
        model = TimesformerForVideoClassification.from_pretrained(config.model_name)
    return model, processor


def eval_spacetime(config: ConfigParser, model_name):
    logger = config.get_logger('Eval')
    logger.info(f'Evaluation started for model: {model_name}')

    # Data transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load dataset and dataloader
    test_path = config.test_path
    logger.info(f'Loading dataset from {test_path}')
    test_dataset = UCF101Dataset(test_path, transform=transform)
    test_dataloader = DataLoader(test_dataset, 1, config.shuffle)
    print('Test dataset:', len(test_dataset), 'samples')

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, processor = load_model(config, model_name, logger)
    model.to(device)

    # Extract embeddings and predictions
    print('[INFO] Model Loaded')
    embeddings, labels, predictions = extract_embeddings_and_predictions(model, processor, test_dataloader, device)

    # Print Classification Metrics
    logger.info('[INFO] Classification Metrics')
    # Compute and display metrics
    compute_metrics(labels, predictions)


# Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to train model")
    parser.add_argument('--config', default=None, help='Path to configuration file')
    parser.add_argument('--model_name',default=None, help='Path to CSV file with dataset information to eval the model')
    parser.add_argument('--name', default=None, help='Name of the experiment (used for saving the model)')
    parser.add_argument('--save_dir', default=None, help='Path to where get the saves file')
    args = parser.parse_args()


    config = ConfigParser(args)
    if args.model_name is None:
        exper_name = config.exper_name
        model_name = f'space_time_{exper_name}_{date.today().strftime("%d-%m-%y")}'
    else:
        model_name = args.model_name
    eval_spacetime(config, model_name)
