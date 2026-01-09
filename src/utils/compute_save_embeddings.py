import csv
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))
import html
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import glob
from PIL import Image
from torchvision import transforms

from src.support_functions.utilis_combination_text_video import (
    extract_videos_embedding,
    extract_text_embeddings_weight,
)
from parse_config import ConfigParser
from src.support_functions.utilis_combination_text_video import (
    load_model_embeddings,
    load_pretrained_text_model_with_embeddings,
)

def add_embedding(dataframe, config, text_encoder, video_encoder, tokenizer, transformer,
                  save_dir, save_name='embeddings', flush_interval=50):
    os.makedirs(save_dir, exist_ok=True)
    row_count = 0
    buffer = []

    with torch.no_grad():
        for index, row in dataframe.iterrows():
            try:
                ID, caption, path, label = row['videoID'], row['caption'], row['path'], row['label']

                # Text embedding
                tokens = tokenizer(caption, padding="max_length", truncation=True,
                                   max_length=config.max_seq_len, return_tensors="pt")
                te = extract_text_embeddings_weight(text_encoder, tokenizer, tokens, config.max_seq_len)
                if isinstance(te, torch.Tensor):
                    te = te.cpu().detach().half().numpy()  # float16
                # Video embedding
                frames = sorted(glob.glob(f"{path}/*.jpg"))[:config.num_frames]
                imgs = [transformer(Image.open(f).convert("RGB")) for f in frames]
                vt = torch.stack(imgs, dim=0).unsqueeze(0).to(video_encoder.device)
                ve = extract_videos_embedding(video_encoder, vt)
                ve = F.adaptive_avg_pool1d(ve.permute(0,2,1), 245).permute(0,2,1)
                ve = ve.cpu().detach().half().numpy()

                buffer.append({
                    'videoID': ID,
                    'caption': caption,
                    'video_path': path,
                    'label': label,
                    'text_embedding': te,
                    'video_embedding': ve,
                })

                row_count += 1
                if row_count % flush_interval == 0:
                    pd.DataFrame(buffer).to_pickle(
                        os.path.join(save_dir, f"{save_name}_{row_count}.pkl"))
                    buffer.clear()
            except Exception as e:
                print(f"[WARNING] Skipping row {index} (ID={row.get('videoID', 'N/A')}): {e}")
                continue  # move to the next row
    if buffer:
        pd.DataFrame(buffer).to_pickle(
            os.path.join(save_dir, f"{save_name}_final.pkl"))

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
    test_path = config.test_path

    train_df = load_dataset(train_csv)
    val_df = load_dataset(val_csv)
    test_df = load_dataset(test_path)


    add_embedding(train_df, config, text_encoder, video_encoder, tokenizer, transform, save_dir= home, save_name = 'Train_embeddings', flush_interval=100)
    add_embedding(val_df, config, text_encoder, video_encoder, tokenizer, transform, save_dir= home, save_name = 'Val_embeddings', flush_interval=100)
    add_embedding(test_df, config, text_encoder, video_encoder, tokenizer, transform,  save_dir= home, save_name = 'Test_embeddings', flush_interval=100)



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