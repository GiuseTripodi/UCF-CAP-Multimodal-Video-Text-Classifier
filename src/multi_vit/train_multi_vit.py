import sys
import os
from datetime import date
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.parse_config import ConfigParser
from src.utils.support_functions import load_dataset
from model.multi_vit import MultimodalSpaceTimeTransformer
from src.trainers.trainer_multimodal import MultimodalTrainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'



def main(config: ConfigParser):
    """
    Main training function.

    Args:
        config: Configuration object
    """
    logger = config.get_logger('Train')
    logger.info(
        f'Training started: multimodal_{config.exper_name}_'
        f'{date.today().strftime("%d-%m-%y")}'
    )

    # Load datasets
    logger.info('Loading datasets...')
    train_loader, val_loader, class_weights = load_dataset(config)
    num_classes = len(class_weights)
    logger.info(f'Loaded {num_classes} classes')

    # Setup model
    logger.info('Initializing model...')
    model = MultimodalSpaceTimeTransformer(
        num_classes=len(class_weights),
        video_model_name=config.video_model_name,
        text_model=config.text_model_name,
        fusion_method=config.fusion_method,
        freeze_video_backbone=config.freeze_video_backbone,
        trainable_layers=config.trainable_layers
    )
    logger.info(f'Model loaded on device: {device}')

    # Setup training components
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs
    )

    # Initialize trainer
    trainer = MultimodalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=config.num_epochs,
        save_dir=config.save_dir,
        logger=logger
    )

    # Train the model
    trainer.train()

    # Print final results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Validation Accuracy: {trainer.best_accuracy:.4f}")
    print(f"Results saved to: {config.save_dir}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multimodal video-text training script")
    parser.add_argument('--name', default='Test', help='Experiment name')
    parser.add_argument(
        '--config',
        default='/Users/user/PycharmProjects/frozen-in-time/configs/ucf-cap.json',
        help='Path to config file'
    )
    parser.add_argument(
        '--save_dir',
        default='/Users/user/PycharmProjects/frozen-in-time/data',
        help='Directory to save checkpoints and results'
    )
    parser.add_argument('--label_experiments', default='ALL', help='Label type for experiments')
    parser.add_argument('--dataset_samples', default=100, type=int, help='Number of dataset samples')

    args = parser.parse_args()
    config = ConfigParser(args)

    main(config)
