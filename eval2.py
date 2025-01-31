import sys
import os
from datetime import date

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import torch

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
from data_loader.lung_pet_ct import DICOMVolumeDataset
import pandas as pd
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, TimesformerForVideoClassification

# Configure and create a logger
logger = logging.getLogger('train')


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
                inputs = processor(inputs, return_tensors="pt")

            inputs = inputs.to(device)

            # Get the embeddings and predicted classes
            outputs = model(inputs)
            logits = outputs.logits

            predicted_labels = torch.argmax(logits, dim=1)  # Get the predicted class

            labels.append(label.numpy())
            predictions.append(predicted_labels.cpu().numpy())

    #embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    return embeddings, labels, predictions


def eval_spacetime(config: ConfigParser, model_name):
    logger = config.get_logger('Eval')
    logger.info(f'Evaluation started for model: {model_name}')

    # Load dataset and dataloader
    test_path = config.test_path
    logger.info(f'Loading dataset from {test_path}')
    dataset = DICOMVolumeDataset(test_path, config)
    test_dataloader = DataLoader(dataset, 1, config.shuffle)
    print('Test dataset:', len(dataset), 'samples')

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = os.path.join(config.save_dir, f'{model_name}')
    #processor = AutoImageProcessor.from_pretrained(model_path)
    processor = None
    model = TimesformerForVideoClassification.from_pretrained(model_path).to(device)
    logger.info(f"Epochs: {config.epochs} - Num Frames: {config.num_frames} - Batch size: {config.batch_size} - Depth: {config.depth} - Heads: {config.num_heads} ")


    # Extract embeddings and predictions
    print('[INFO] Model Loaded')
    embeddings, labels, predictions = extract_embeddings_and_predictions(model, processor, test_dataloader, device)

    # Print Classification Metrics
    print('[INFO] Classification Metrics')
    logger.info('[INFO] Classification Metrics')

    accuracy = (predictions == labels).sum().item() / len(labels)  # Accuracy calculation
    print(f'Accuracy: {accuracy * 100:.2f}%')  # Print accuracy as percentage
    print(f'Precision: {precision_score(labels, predictions, average="weighted")}')
    print(f'Recall: {recall_score(labels, predictions, average="weighted")}')
    print(f'F1 Score: {f1_score(labels, predictions, average="weighted")}')
    print(f'Classification Report: \n{classification_report(labels, predictions)}')
    logger.info(f'Accuracy: {accuracy * 100:.2f}%')
    logger.info(f'Precision: {precision_score(labels, predictions, average="weighted")}')
    logger.info(f'Recall: {recall_score(labels, predictions, average="weighted")}')
    logger.info(f'F1 Score: {f1_score(labels, predictions, average="weighted")}')
    logger.info(f'Classification Report: \n{classification_report(labels, predictions)}')



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
