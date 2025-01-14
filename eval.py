import argparse
import sys
import os
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
def load_model(model_path, num_classes=10):
    model = SpaceTimeTransformer(
        img_size=224,
        num_frames=8,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        attention_style='frozen-in-time'
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set model to evaluation mode
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


# Define a function to plot embeddings
def plot_embeddings(embeddings, labels, method='pca'):
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError("Unsupported method. Use 'pca' or 'tsne'.")

    reduced_embeddings = reducer.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Class Labels')
    plt.title(f"Embeddings Visualization ({method.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()


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
def evaluation(test_path, model_path):
    # Data transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset and dataloader
    test_dataset = UCF101Dataset(test_path, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(model_path, num_classes=13).to(device)

    # Extract embeddings and predictions
    print('[INFO] Model Loaded')
    embeddings, true_labels, pred_labels = extract_predictions(model, test_dataloader, device)

    # Compute and display metrics
    compute_metrics(true_labels, pred_labels)

    # Plot embeddings
    #plot_embeddings(embeddings, true_labels, method='tsne')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to train model")
    parser.add_argument('--data', help='path to data dir used to train the model')
    parser.add_argument('--model_path', default=None, help='Path where to save the trained models')
    parser.add_argument('--name', default=None, help='Name of the experiments, way of saving the model ')
    parser.add_argument('--log', default=None, help="Path to where the logs are saved")
    parser.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args = parser.parse_args()

    #setup_logging(args.log)

    logger.info("Training started")
    evaluation(args.data, args.model_path)