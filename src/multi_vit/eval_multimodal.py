import argparse
import os
import sys
import torch
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data_loader.ucf_cap_loader import UCF101Dataset, collate_video_batch
from model.multi_vit import MultimodalSpaceTimeTransformer
from src.utils.parse_config import ConfigParser
from torch.utils.data import DataLoader


def evaluate(config: ConfigParser, checkpoint_path: str) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = UCF101Dataset(
        config.test_path,
        sampling_method=config.sampling_method,
        num_frames=config.num_frames,
        img_size=config.img_size
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_video_batch
    )

    model = MultimodalSpaceTimeTransformer(
        num_classes=config.num_classes,
        video_model_name=config.video_model_name,
        text_model=config.text_model_name,
        fusion_method=config.fusion_method,
        freeze_video_backbone=True,
        trainable_layers=config.trainable_layers
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for videos, texts, labels in dataloader:
            videos = videos.to(device)
            logits = model(videos, texts)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(classification_report(all_labels, all_preds, digits=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate multimodal model on test set')
    parser.add_argument(
        '--config',
        default='/Users/user/PycharmProjects/frozen-in-time/configs/ucf-cap.json',
        help='Path to config file'
    )
    parser.add_argument(
        '--save_dir',
        default='/Users/user/PycharmProjects/frozen-in-time/data',
        help='Directory containing UcfCap and checkpoints'
    )
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='Path to checkpoint file (best_model_.pth)'
    )
    parser.add_argument('--name', default='Eval', help='Experiment name')

    args = parser.parse_args()
    config = ConfigParser(args)
    evaluate(config, args.checkpoint)

