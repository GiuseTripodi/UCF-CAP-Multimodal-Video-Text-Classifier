import sys
import os
import argparse
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.parse_config import ConfigParser
from model.multi_vit import MultimodalSpaceTimeTransformer

from data_loader.ucf_cap_loader import UCF101Dataset
from src.multi_vit.train_multi_vit import multimodal_collate_fn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate(model, val_loader, device):
    """Run evaluation loop and collect metrics."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        val_loop = tqdm(val_loader, desc="Evaluating", leave=True)
        for videos, texts, labels in val_loop:
            videos = videos.to(device)
            labels = labels.to(device)

            logits = model(videos, texts)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(all_labels, all_preds))

    return acc, precision, recall, f1, cm

def load_test_dataset(config:ConfigParser):
    csv_train_file_path = config.test_path
    dataset_test = UCF101Dataset(csv_train_file_path, sampling_method='uniform')
    test_loader = DataLoader(
        dataset_test, batch_size=config.batch_size, shuffle=True,
        num_workers=0, collate_fn=multimodal_collate_fn
    )
    return test_loader

def main(config: ConfigParser, model_name: str):
    print(f"🔍 Loading validation dataset using config from: {config.config}")
    test_loader = load_test_dataset(config)

    print(f"📦 Loading model from: {os.path.join(config.save_dir, model_name)}")
    model = MultimodalSpaceTimeTransformer(
        num_classes=12,
        text_model='distilbert-base-uncased',
        fusion_method='concat'
    ).to(device)

    model.load_state_dict(torch.load(os.path.join(config.save_dir, model_name), map_location=device))
    print("✅ Model weights loaded successfully!")


    print("🚀 Starting evaluation...")
    results = evaluate(model, test_loader, device)

    # Optionally, save results
    np.savez('multimodal_validation_results.npz',
             loss=results[0],
             accuracy=results[1],
             precision=results[2],
             recall=results[3],
             f1=results[4],
             confusion_matrix=results[5])

    print("\n✅ Evaluation complete! Results saved to multimodal_validation_results.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MultimodalSpaceTimeTransformer on validation set")
    parser.add_argument('--name', default='Evaluating Multi VIT', help='Experiment name')
    parser.add_argument('--config', default='/Users/user/PycharmProjects/frozen-in-time/configs/ucf-cap.json', help='Config file path')
    parser.add_argument('--model_name', default='best_multimodal_model.pth', help='Path to trained model weights')
    parser.add_argument('--save_dir', default='/Users/user/PycharmProjects/frozen-in-time/data', help='Save directory')
    args = parser.parse_args()

    config = ConfigParser(args)
    main(config, args.model_name)
