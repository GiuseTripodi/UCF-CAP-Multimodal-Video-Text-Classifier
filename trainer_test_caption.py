import sys
import os
import collections


from parse_config import ConfigParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))
from trainer import Trainer
import argparse
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from model.video_transformer import *
from data_loader.ucf_cap_dataset import *
import logging
from logger import setup_logging
import model.metric as module_metric



# Configure and create a logger
logger = logging.getLogger('train')

def training():
    csv_train_file = '/Users/user/PycharmProjects/frozen-in-time/data/UcfCap/train_dataset.csv'
    csv_val_file = '/Users/user/PycharmProjects/frozen-in-time/data/UcfCap/val_dataset.csv'


    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    logger.info(f'Dataset loaded from {csv_train_file}')
    dataset = UCF101Dataset(csv_train_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    dataset_val = UCF101Dataset(csv_val_file, transform=transform)
    valid_data_loader = DataLoader(dataset_val, batch_size=8, shuffle=True)
    print('Train dataset: ', [len(x) for x in dataloader], ' samples')


    model = SpaceTimeTransformer(
        img_size=224,         # Resize frames to 224x224
        num_frames=8,         # Use 8 frames per video
        in_chans=3,           # RGB channels
        num_classes=10,
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

    # Define Loss, Optimizer, and Metrics
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    metrics_list = [
        "t2v_metrics",
        "v2t_metrics"
     ]
    metrics = [getattr(module_metric, met) for met in metrics_list]


    trainer = Trainer(
        model=model,
        loss=criterion,
        metrics=metrics,
        optimizer=optimizer,
        config=config,
        data_loader=dataloader,
        valid_data_loader=valid_data_loader
    )

    # Start Training
    trainer.train()

    # Save the model's state dictionary
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    logger.info(f"Model saved to {save_path}")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('--data', help='path to data dir used to train the model')
    args.add_argument('--save', default=None, help='Path where to save the trained models')
    args.add_argument('--name', default=None, help='Name of the experiments, way of saving the model ')
    args.add_argument('--log', default=None, help="Path to where the logs are saved")
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    #args = parser.parse_args()

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size')),
    ]
    config = ConfigParser(args, options, test=True)

    logger.info("Training started")
    training()