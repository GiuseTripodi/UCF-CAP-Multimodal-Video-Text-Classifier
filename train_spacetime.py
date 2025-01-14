import sys
import os
import collections

from trainer.trainer_video import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))
import argparse
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from model.video_transformer import *
from data_loader.ucf_cap_dataset import *
import logging
from logger import setup_logging
import model.metric as module_metric
from sklearn.metrics import accuracy_score



# Configure and create a logger
logger = logging.getLogger('train')
# Define a simple function to compute accuracy
def compute_metrics(p):
    preds, labels = p
    preds = torch.argmax(torch.tensor(preds), dim=1)  # Get class predictions
    accuracy = accuracy_score(labels, preds)
    return {"accuracy": accuracy}

def training(data, save, name):
    csv_train_file_path = os.path.join(data, 'train_dataset.csv')
    csv_val_file_path = os.path.join(data, 'val_dataset.csv')


    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    logger.info(f'Training dataset loaded from {csv_train_file_path}')
    logger.info(f'Validation dataset loaded from {csv_val_file_path}')
    dataset_train = UCF101Dataset(csv_train_file_path, transform=transform)
    train_dataloader = DataLoader(dataset_train, batch_size=8, shuffle=True)
    dataset_val = UCF101Dataset(csv_val_file_path, transform=transform)
    valid_dataloader = DataLoader(dataset_val, batch_size=8, shuffle=True)


    model = SpaceTimeTransformer(
        img_size=224,         # Resize frames to 224x224
        num_frames=8,         # Use 8 frames per video
        in_chans=3,           # RGB channels
        num_classes=13,
        embed_dim=768,
        depth=12,
        num_heads=12,
        attention_style='frozen-in-time'
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total Trainable Parameters: {total_params}")
    logger.info(f"Total Trainable Parameters: {total_params}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Initialize Trainer class and start training
    save_path = f'{save}/spacetime_transformer_{name}'

    trainer = Trainer(model, train_dataloader, valid_dataloader, criterion, optimizer, device, save_path, num_epochs=1)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to train model")
    parser.add_argument('--data', help='path to data dir used to train the model')
    parser.add_argument('--save', default=None, help='Path where to save the trained models')
    parser.add_argument('--name', default=None, help='Name of the experiment, used for saving the model')
    parser.add_argument('--log', default=None, help="Path to where the logs are saved")
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log)

    # Start the training
    logger.info("Training started")
    training(args.data, args.save, args.name)