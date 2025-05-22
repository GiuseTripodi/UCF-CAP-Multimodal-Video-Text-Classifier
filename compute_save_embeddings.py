import csv
import sys
import os
from datetime import date

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
import argparse
import logging
import numpy as np
import glob
from PIL import Image
from data_loader.ucf_cap_dataset import UCF101Dataset
from model.MLP_classifier import MLPClassifier
from utils.utilis_combination_text_video import (
    extract_text_embeddings,
    extract_videos_embedding,
    load_text_encoder,
    load_model_embeddings,
    project_text_video, load_pretrained_text_model, extract_text_embeddings_weight,
    load_pretrained_text_model_with_embeddings,
)
from parse_config import ConfigParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

def add_embedding(dataframe, config, text_encoder, video_encoder, tokenizer, transformer,
                  save_dir, save_name='embeddings', flush_interval=100):
    """
    Extract embeddings and every flush_interval rows:
     - Append to disk CSV / pickle
     - Clear in-memory lists
    """
    # Prepare lists and counters
    ids, captions, paths, labels = [], [], [], []
    text_embs, video_embs = [], []
    row_count = 0
    today = date.today().strftime("%d-%m-%y")

    # Ensure save_dir exists
    #os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "UcfCap", f"{save_name}_{today}.csv")
    pkl_path = os.path.join(save_dir, "UcfCap", f"{save_name}_{today}.pkl")

    # If CSV exists already (from previous flush), load it and track how many are already saved
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        row_count = len(existing)
    else:
        # write header if new
        pd.DataFrame([], columns=['videoID','caption','video_path','label','text_embedding','video_embedding']
                    ).to_csv(csv_path, index=False)

    with torch.no_grad():
        for index, row in dataframe.iterrows():
            ID, caption, path, label = row['videoID'], row['caption'], row['path'], row['label']
            # 1) Text embedding
            tokens = tokenizer(caption,
                               padding="max_length", truncation=True,
                               max_length=config.max_seq_len, return_tensors="pt")
            # 1) Text embedding
            te = extract_text_embeddings_weight(text_encoder, tokenizer, tokens, config.max_seq_len)
            if isinstance(te, torch.Tensor):
                te = te.cpu().detach().numpy()
            # now te is a NumPy array
            te = te.tolist()

            # 2) Video embedding
            frames = sorted(glob.glob(f"{path}/*.jpg"))[:config.num_frames]
            imgs = [transformer(Image.open(f).convert("RGB")) for f in frames]
            vt = torch.stack(imgs, dim=0).unsqueeze(0).to(video_encoder.device)
            ve = extract_videos_embedding(video_encoder, vt)
            ve = ve.cpu().detach()
            ve = ve.permute(0,2,1)                       # (N_frames, dim, tokens)
            ve = F.adaptive_avg_pool1d(ve, 245)         # (N_frames, dim, 245)
            ve = ve.permute(0,2,1).numpy().tolist()     # back to list

            # 3) Collect
            ids.append(ID); captions.append(caption)
            paths.append(path); labels.append(label)
            text_embs.append(te); video_embs.append(ve)

            row_count += 1

            # 4) Flush to disk every flush_interval
            if row_count % flush_interval == 0:
                df_flush = pd.DataFrame({
                    'videoID': ids,
                    'caption': captions,
                    'video_path': paths,
                    'label': labels,
                    'text_embedding': text_embs,
                    'video_embedding': video_embs,
                })
                # append to CSV/pickle
                df_flush.to_csv(csv_path, mode='a', header=False, index=False)
                # for pickle, you could append but here we rewrite entire pkl
                full_df = pd.read_csv(csv_path)
                full_df.to_pickle(pkl_path)

                # clear buffers
                ids.clear(); captions.clear(); paths.clear(); labels.clear()
                text_embs.clear(); video_embs.clear()

    # 5) Final flush of any remaining
    if ids:
        df_flush = pd.DataFrame({
            'videoID': ids,
            'caption': captions,
            'video_path': paths,
            'label': labels,
            'text_embedding': text_embs,
            'video_embedding': video_embs,
        })
        df_flush.to_csv(csv_path, mode='a', header=False, index=False)
        full_df = pd.read_csv(csv_path)
        full_df.to_pickle(pkl_path)

    print(f"Finished. Embeddings saved to:\n  {csv_path}\n  {pkl_path}")


def load_dataset(csv_file):
    data = []
    # Read and process the CSV file
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        for line in reader:
            id, caption, path, label = line
            data.append((id, caption, path, label))

    # Create the DataFrame
    return pd.DataFrame(data, columns=['videoID','caption', 'path', 'label'])


def main(config: ConfigParser, model_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger = config.get_logger('TrainMLP')

    # Load encoders
    #tokenizer, text_encoder = load_text_encoder()
    home = "/mnt/iusers01/mace01/t08341gt/UCF_cap_mh/data"
    #home = "/Users/user/PycharmProjects/frozen-in-time/data"
    tokenizer, text_encoder = load_pretrained_text_model_with_embeddings(f'{home}/models/weights_multiclass_31-03-25_bert_training.h5')
    video_encoder, _ = load_model_embeddings(config, model_name, logger)

    #text_encoder.to(device)
    video_encoder.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #load the dataset
    train_csv, val_csv = config.train_path
    train_df = load_dataset(train_csv)
    val_df = load_dataset(val_csv)
    test_path = config.test_path
    test_df = load_dataset(test_path)



    #add_embedding(train_df, config, text_encoder, video_encoder, tokenizer, transform, save_dir= home, save_name = 'Train_embeddings')
    add_embedding(val_df, config, text_encoder, video_encoder, tokenizer, transform, save_dir= home, save_name = 'Val_embeddings')
    add_embedding(test_df, config, text_encoder, video_encoder, tokenizer, transform,  save_dir= home, save_name = 'Test_embeddings')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to train MLP model with validation")
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--model_name', required=True, help='Name of the experiment')
    parser.add_argument('--save_dir', default=None, help='Path to where get the saves file')
    parser.add_argument('--name', default=None, help='Name of the experiment (used for saving the model)')
    args = parser.parse_args()

    config = ConfigParser(args)
    model_name = args.model_name
    expt_name = args.name

    main(config, model_name)