import ast
from os.path import join

import pandas as pd
import os
import csv
import re

from sklearn.utils import shuffle
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import glob
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import glob
from PIL import Image
import os
import re
import csv
from sklearn.model_selection import train_test_split
import numpy as np
import torchvision.io as io

def create_csv_splits(home_path):
    # Paths to the UCF101 dataset
    data_dir = f'{home_path}/data/UcfCap/YouTubeClips'  # Replace with your UCF101 frames directory
    output_dir = f'{home_path}/data/UcfCap/'  # Directory to save CSV files
    mapping_path = f'{home_path}/data/UcfCap/captions/youtube_mapping.txt'

    # Load mapping
    mapping = {}
    with open(mapping_path, "r") as file:
        for line in file:
            key, value = line.strip().split()  # Split by whitespace
            mapping[key] = value  # Store in dictionary


    # Get the list of class names (folder names in the dataset)
    videos_folder_frame = sorted(os.listdir(data_dir))

    # Collect video paths and their labels
    data_entries = []
    for video_fold in videos_folder_frame:
        video_path = os.path.join(data_dir, video_fold)
        match = re.search(r'([a-zA-Z]+)\d', video_fold)
        if match and os.path.isdir(video_path):  # Ensure it's a directory and matches the pattern
            label = match.group(1)
            data_entries.append((video_path, label, mapping[video_fold]))

    df_mapping = pd.DataFrame(data_entries, columns=["video_path", "label", "videoID"])
    def load_txt_splitting(path):
        with open(path, "r", encoding="utf-8") as file:
            data = [line.strip().split(" ", 1) for line in file]  # Split only on the first space

        # Convert to DataFrame
        df = pd.DataFrame(data, columns=["videoID", "caption"])
        return df

    # Load test, train, val txt file and merge it based on video id
    df_test = load_txt_splitting(f'{home_path}/data/UcfCap/captions/sents_test_lc_nopunc.txt')
    df_val = load_txt_splitting(f'{home_path}/data/UcfCap/captions/sents_val_lc_nopunc.txt')
    df_train = load_txt_splitting(f'{home_path}/data/UcfCap/captions/sents_train_lc_nopunc.txt')

    # Merge the two datasets on 'videoID'
    merged_df_test = pd.merge(df_test, df_mapping, on="videoID", how="inner")
    merged_df_val = pd.merge(df_val, df_mapping, on="videoID", how="inner")
    merged_df_train = pd.merge(df_train, df_mapping, on="videoID", how="inner")

    # Save the merged dataset
    merged_df_test.to_csv(join(output_dir, "test_dataset.csv"), index=False)
    merged_df_val.to_csv(join(output_dir, "val_dataset.csv"), index=False)
    merged_df_train.to_csv(join(output_dir, "train_dataset.csv"), index=False)



class UCF101Dataset(Dataset):
    def __init__(self, csv_file, transform=None, num_frames=8, num_samples=100,):
        self.num_frames = num_frames
        self.transform = transform

        # Read and process the CSV file
        if csv_file.split('.')[-1] == 'csv':
            df = pd.read_csv(csv_file)
        else:
            df = pd.read_pickle(csv_file)
        print(df.columns)

        # Get all unique classes
        classes = df['label'].unique()
        print(classes)
        num_classes = len(classes)
        samples_per_class = num_samples // num_classes

        # Sample evenly from each class
        df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(min(len(x), samples_per_class), random_state=42))
        self.data = shuffle(df, random_state=42)
        # print number of classes per sample
        # Count how many samples per class
        print(f"Labels distributions for: {csv_file} \\n {self.data['label'].value_counts().sort_index()}")


        # Initialize a LabelEncoder to convert string labels to integer indices
        self.label_encoder = LabelEncoder()
        # Fit the label encoder to the labels
        self.label_encoder.fit([label for label in self.data['label']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx_, caption, path, label = self.data.iloc[idx]['videoID'], self.data.iloc[idx]['caption'], self.data.iloc[idx]['video_path'], self.data.iloc[idx]['label']
        text_embedding = []
        video_embedding = None

        if len(self.data.columns) > 4:
            text_embedding = self.data.iloc[idx]['text_embedding']
            video_embedding = self.data.iloc[idx]['video_embedding']

        frames = sorted(glob.glob(os.path.join(path, '*.jpg')))
        selected_frames = frames[:self.num_frames] # Choose first N frames
        images = [Image.open(frame).convert("RGB") for frame in selected_frames]
        # Check pixel values for the first image
        if self.transform:
            images = [self.transform(img) for img in images]

        try:
            video_tensor = torch.stack(images, dim=0)  # Shape: [num_frames, C, H, W]
        except:
            video_tensor = torch.tensor(np.zeros((self.num_frames, 3, 224, 224)), dtype=torch.float32)
            print(f"Using placeholder volume for: {idx}")

        # Convert label to integer using label_encoder
        label_idx = self.label_encoder.transform([label])[0]  # Convert string label to integer index

        return video_tensor, caption, label_idx, text_embedding, video_embedding  # Return the label index as an integer tensor


if __name__ == '__main__':
    # Run the function
    home_path = '/Users/user/PycharmProjects/frozen-in-time'
    #home_path = '/mnt/iusers01/mace01/t08341gt/UCF_cap_mh'
    create_csv_splits(home_path)

    '''
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    csv_file = '/Users/user/PycharmProjects/frozen-in-time/data/UcfCap/val_dataset.csv'

    dataset = UCF101Dataset(csv_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    '''