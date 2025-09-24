import sys
import os
from datetime import date

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import torch

from data_loader.ucf_cap_dataset import UCF101Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
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
def compute_metrics(labels, predictions, logger):
    logger.info('[INFO] Classification Metrics')
    accuracy = (predictions == labels).sum().item() / len(labels)  # Accuracy calculation
    logger.info(f'Accuracy: {accuracy * 100:.2f}%')
    logger.info(f'Precision: {precision_score(labels, predictions, average="weighted")}')
    logger.info(f'Recall: {recall_score(labels, predictions, average="weighted")}')
    logger.info(f'F1 Score: {f1_score(labels, predictions, average="weighted")}')
    logger.info(f'Classification Report: \n{classification_report(labels, predictions)}')



# Define a function to extract embeddings and predictions
def extract_embeddings_and_predictions(model, processor, dataloader, device):
    embeddings = []
    labels = []
    predictions = []
    with torch.no_grad():
        for inputs, captions, label, *other_info in dataloader:
            inputs = inputs.to(device)
            if processor:
                inputs = inputs.squeeze(0)  # Removes the batch dimension
                inputs = processor(list(inputs), return_tensors="pt")

            # Get the embeddings and predicted classes
            outputs = model(inputs)
            if config.modality == 1:
                outputs = outputs.logits
            predicted_labels = torch.argmax(outputs, dim=1)  # Get the predicted class

            #embeddings.append(embedding.cpu().numpy())
            labels.append(label.numpy())
            predictions.append(predicted_labels.cpu().numpy())

    #embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    return embeddings, labels, predictions


def load_model(config: ConfigParser, model_name, logger):
    if config.modality == 0:
        model_path = os.path.join(config.save_dir, f'{model_name}')
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
        model.load_state_dict(torch.load(model_path))
        processor = None

    elif config.modality == 1:
        # Use model pre_trained and the fine_tuned
        model_path = os.path.join(config.save_dir, f'{model_name}')
        logger.info(f"[INFO] Loaded fine_tuned model from: {model_path}")
        model = TimesformerForVideoClassification.from_pretrained(model_path)
        # processor = AutoImageProcessor.from_pretrained(model_path)
        processor = None
        print(model.config)

    return model, processor


def eval_spacetime(config: ConfigParser, model_name):
    logger = config.get_logger('Eval')
    logger.info(f'Evaluation started for model: {model_name}')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

    # Compute and display metrics
    compute_metrics(labels, predictions, logger)


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
