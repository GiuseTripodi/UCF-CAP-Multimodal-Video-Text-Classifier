import sys
import os
from datetime import date

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import torch

from data_loader.ucf_cap_dataset import UCF101Dataset
from model.MLP_classifier import MLPClassifier

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
def compute_metrics(labels, predictions, logger):
    logger.info('[INFO] Classification Metrics')
    accuracy = (predictions == labels).sum().item() / len(labels)  # Accuracy calculation
    logger.info(f'Accuracy: {accuracy * 100:.2f}%')
    logger.info(f'Precision: {precision_score(labels, predictions, average="weighted")}')
    logger.info(f'Recall: {recall_score(labels, predictions, average="weighted")}')
    logger.info(f'F1 Score: {f1_score(labels, predictions, average="weighted")}')
    logger.info(f'Classification Report: \n{classification_report(labels, predictions)}')



# Define a function to extract embeddings and predictions
def train_model_MLP(text_encoder, video_encoder, classifier, dataloader, criterion, optimizer, device):
    text_encoder.eval()
    video_encoder.eval()
    classifier.train()

    with torch.no_grad():
        for inputs, labels, *other_info in tqdm(dataloader, desc="Training Epoch"):
            inputs, labels = inputs.to(device), labels.to(device)

            #TODO implementare embedding computation with the loaded model
            text_embeddings = None
            image_embeddings = None

            # Al momento gli embedding sono solo concatenati (come?)
            combined_embeddings = torch.cat((text_embeddings, image_embeddings), dim=1)
            #Todo Decidere se usare la similarità del coseno, forse se non alleniamo il modello basandoci sulla similarità poco senso usare cos_sim
            '''
            similarity = compute_similarity(text_embeddings, image_embeddings)

            # Concatenate embeddings with similairyt, high similarity 
            combined_embeddings = torch.cat((text_embeddings, image_embeddings, similarity), dim=1)

            '''

            outputs = classifier(combined_embeddings)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Load Text Encoder
def load_text_encoder(model_name='bert-base-uncased'):
    #TODO implement loading of the model
    return None

def load_video_encoder(config: ConfigParser, model_name, logger):
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


def run_training(config: ConfigParser, model_name):
    logger = config.get_logger('TrainMLP')
    logger.info(f'MLPtraining_started: {model_name}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load encoders
    text_encoder = load_text_encoder()
    image_encoder = load_video_encoder()

    text_encoder.to(device)
    image_encoder.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #TODO modificare data loader per far restituire anche il testo (forse creare un nuovo data loader)
    csv_train_file_path, csv_val_file_path = config.train_path
    logger.info(f'Loading dataset from {csv_train_file_path}')
    dataset_train = UCF101Dataset(csv_train_file_path, transform=transform)
    train_dataloader = DataLoader(dataset_train, config.batch_size, config.shuffle)
    dataset_val = UCF101Dataset(csv_val_file_path, transform=transform)
    val_dataloader = DataLoader(dataset_val, config.batch_size, config.shuffle)
    # Load test file
    test_path = config.test_path
    test_dataset = UCF101Dataset(test_path, transform=transform)
    test_dataloader = DataLoader(test_dataset, 1, config.shuffle)

    # Initialize MLP classifier
    #TODO check dimension combination and mofigy
    input_dim = 500
    hidden_dim = 512
    num_classes = config.num_classes
    classifier = MLPClassifier(input_dim, hidden_dim, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=config.learning_rate)


    # Extract embeddings and predictions
    print('[INFO] Started training')
    #TODO modify funciton to return the trained MLP
    train_model_MLP(train_dataloader, text_encoder, image_encoder, classifier, criterion, optimizer, device)

    # Compute and display metrics
    #TODO modify function to use the trained MLP on test and get results
    #compute_metrics(labels, predictions, logger)


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
