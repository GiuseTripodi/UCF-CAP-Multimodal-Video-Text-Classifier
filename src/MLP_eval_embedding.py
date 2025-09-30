import sys
import os
from datetime import date
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import torch
import torch.nn.functional as F

from data_loader.ucf_cap_dataset import UCF101Dataset
from model.MLP_classifier import MLPClassifier
from utils.utilis_combination_text_video import extract_text_embeddings, extract_text_embeddings_weight, \
    load_text_encoder, load_pretrained_text_model, extract_videos_embedding, load_model_embeddings, project_text_video, \
    load_pretrained_text_model_with_embeddings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))
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
from transformers import AutoImageProcessor, TimesformerForVideoClassification, DistilBertTokenizer, TFDistilBertModel
import tensorflow as tf


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
def evaluate_model(classifier, dataloader, device, config, logger):
    classifier.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, caption, labels, text_embeddings, video_embeddings, *_ in tqdm(dataloader, desc=f"Epoch eval"):
            inputs, labels, text_embeddings, video_embeddings, = inputs.to(device), labels.to(device), text_embeddings.to(device), video_embeddings.to(device)

            text_embeddings = text_embeddings.squeeze(1)
            video_embeddings = video_embeddings.squeeze(1)

            combined_embedding = torch.cat((text_embeddings, video_embeddings), dim=-1)
            combined_embedding = combined_embedding.mean(dim=1).float()

            # Forward pass
            outputs = classifier(combined_embedding)
            predictions = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

        # Compute and log metrics
        compute_metrics(torch.tensor(all_labels), torch.tensor(all_predictions), logger)


# Load Text Encoder




def run_eval(config: ConfigParser, model_name):
    logger = config.get_logger('TrainMLP')
    logger.info(f'MLP_eval_started: {model_name}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load test file
    test_path = "/Users/user/PycharmProjects/frozen-in-time/data/UcfCap/Test_embeddings_all.pkl"
    test_dataset = UCF101Dataset(test_path, transform=transform)
    test_dataloader = DataLoader(test_dataset, 1, config.shuffle)

    # Initialize MLP classifier
    #TODO check dimension combination and mofigy
    input_dim = 1536
    hidden_dim = 256
    num_classes = 12
    model_path = "/Users/user/PycharmProjects/frozen-in-time/data/models/ciccio_26-09-25_final_MLP.pth"
    classifier = MLPClassifier(input_dim, hidden_dim, num_classes).to(device)
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    logger.info(f'Loaded model from {model_path}')


    # Evaluate the model
    evaluate_model(classifier, test_dataloader, device, config, logger)


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
    run_eval(config, model_name)
