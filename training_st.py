import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import argparse
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from model.video_transformer import *
from data_loader.ucf_cap_dataset import *
import logging
from logger import setup_logging


# Configure and create a logger
logger = logging.getLogger('train')

def training(data, save, name):
    save_path = f'{save}/spacetime_transformer_{name}.pth'
    csv_train_file = data


    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    logger.info(f'Dataset loaded from {csv_train_file}')
    dataset = UCF101Dataset(csv_train_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    #print('Train dataset: ', [len(x) for x in dataloader], ' samples')


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
    logger.info(f"Device model: {device}")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    logger.info(model)

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels, *other_info in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")
        logger.info(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")

    # Save the model's state dictionary
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    logger.info(f"Model saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to train model")
    parser.add_argument('--data', help='path to data dir used to train the model')
    parser.add_argument('--save', default=None, help='Path where to save the trained models')
    parser.add_argument('--name', default=None, help='Name of the experiments, way of saving the model ')
    parser.add_argument('--log', default=None, help="Path to where the logs are saved")
    parser.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args = parser.parse_args()

    setup_logging(args.log)

    logger.info("Training started")
    training(args.data, args.save, args.name)