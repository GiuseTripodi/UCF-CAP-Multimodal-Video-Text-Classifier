import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from data_loader.ucf_cap_dataset import UCF101Dataset
from model.video_transformer import SpaceTimeTransformer

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

# Define a function to extract embeddings
def extract_embeddings(model, dataloader, device):
    embeddings = []
    labels = []
    with torch.no_grad():
        for inputs, label in dataloader:
            inputs = inputs.to(device)

            # Get the embeddings
            features = model.forward_features(inputs)  # Access embeddings before classification head
            embeddings.append(features.cpu().numpy())
            labels.append(label.numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    return embeddings, labels

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

# Main function
if __name__ == '__main__':
    # Paths
    model_path = '/data/models/spacetime_transformer_ucf101.pth'
    csv_test_file = '/Users/user/PycharmProjects/frozen-in-time/data/UcfCap/test_dataset.csv'

    # Data transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset and dataloader
    test_dataset = UCF101Dataset(csv_test_file, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(model_path, num_classes=10).to(device)

    # Extract embeddings
    print('[INFO] Model Loaded')
    embeddings, labels = extract_embeddings(model, test_dataloader, device)

    print(labels)
    print(embeddings)

    # Plot embeddings
    plot_embeddings(embeddings, labels, method='tsne')
