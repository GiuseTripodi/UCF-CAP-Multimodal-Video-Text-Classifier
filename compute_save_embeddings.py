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

def add_embedding(dataframe, config: ConfigParser, text_encoder, video_encoder, tokenizer, tranformer, save_name = 'embeddings'):

    text_embeddings = []
    video_embeddings = []
    captions = []
    labels = []

    with torch.no_grad():
        for index, row in dataframe.iterrows():
            caption, path, label= row['caption'], row['path'], row['label']
            caption_tokens = tokenizer(caption, padding="max_length", truncation=True, max_length=config.max_seq_len,
                                       return_tensors="pt")

            frames = sorted(glob.glob(os.path.join(path, '*.jpg')))
            selected_frames = frames[:config.num_frames]  # Choose first N frames
            images = [Image.open(frame).convert("RGB") for frame in selected_frames]
            images = [tranformer(img) for img in images]
            video_tensor = torch.stack(images, dim=0).unsqueeze(0)  # Shape: [num_frames, C, H, W]

            # text_embeddings = extract_text_embeddings(text_encoder, tokenizer, caption_tokens, config.max_seq_len)
            try:
                text_embedding = extract_text_embeddings_weight(text_encoder, tokenizer, caption_tokens, config.max_seq_len)
                text_embeddings.append(text_embedding)

            except:
                print(f'Error with the generation of the text embedding for row: {index}')

            # text_embeddings = extract_text_embeddings(text_encoder, tokenizer, caption_tokens, config.max_seq_len)
            try:
                image_embedding = extract_videos_embedding(video_encoder, video_tensor)
                video_embeddings.append(image_embedding)
                del image_embedding
            except:
                print(f'Error with the generation of the video embedding for row: {index}')
            captions.append(caption)
            labels.append(label)

    df = pd.DataFrame({
        'caption': captions,
        'label': labels,
        'text_embedding': text_embeddings,
        'video_embedding': video_embeddings,
    })

    df.to_csv(f'/Users/user/PycharmProjects/frozen-in-time/data/UcfCap/{save_name}_{date.today().strftime("%d-%m-%y")}.csv', index=False)



def load_dataset(csv_file):
    data = []

    # Read and process the CSV file
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        for line in reader:
            id, caption, path, label = line
            data.append((caption, path, label))

    data = data[:50]
    # Create the DataFrame
    return pd.DataFrame(data, columns=['caption', 'path', 'label'])


def main(config: ConfigParser, model_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger = config.get_logger('TrainMLP')


    # Load encoders
    #tokenizer, text_encoder = load_text_encoder()
    tokenizer, text_encoder = load_pretrained_text_model_with_embeddings('/Users/user/PycharmProjects/frozen-in-time/data/models/weights_multiclass_31-03-25_bert_training.h5')
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

    add_embedding(train_df, config, text_encoder, video_encoder, tokenizer, transform, save_name = 'Train_embeddings')
    add_embedding(val_df, config, text_encoder, video_encoder, tokenizer, transform, save_name = 'Val_embeddings')

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