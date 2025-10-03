from os.path import join
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import re
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import glob, os

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


def load_video(path, num_frames=16, out_size=(16, 224, 224)):
    # 1. Load frames
    frames = sorted(glob.glob(os.path.join(path, '*.jpg')))

    transform = transforms.ToTensor()
    images = [transform(Image.open(f).convert("RGB")) for f in frames]  # each (C,H,W)

    # 2. Stack into (D, C, H, W)
    video = torch.stack(images)  # (D, C, H, W)

    # 3. Rearrange to (1, C, D, H, W) for interpolate
    video = video.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, D, H, W)

    # 4. Interpolate to fixed (D,H,W)
    target_shape = (out_size[0], out_size[1], out_size[2])  # (D,H,W)
    video = F.interpolate(video, size=target_shape, mode="trilinear", align_corners=False)

    # 5. Rearrange back to model input: (1, D, C, H, W)
    video = video.squeeze(0).permute(1, 0, 2, 3)

    return video  # (frames=D, channels=C, H, W)



class UCF101Dataset(Dataset):
    def __init__(self, csv_file, transform=None, num_frames=8, num_samples=100,):
        self.num_frames = num_frames
        self.transform = transform

        # Read and process the CSV file
        if csv_file.split('.')[-1] == 'csv':
            df = pd.read_csv(csv_file)
        else:
            df = pd.read_pickle(csv_file)
        self.data = shuffle(df, random_state=42)

        # Initialize a LabelEncoder to convert string labels to integer indices
        self.label_encoder = LabelEncoder()
        # Fit the label encoder to the labels
        self.label_encoder.fit([label for label in self.data['label']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx_, caption, path, label = self.data.iloc[idx]['videoID'], self.data.iloc[idx]['caption'], self.data.iloc[idx]['video_path'], self.data.iloc[idx]['label']
        text_embedding = []
        video_embedding = []

        if len(self.data.columns) > 4:
            text_embedding = np.array(self.data.iloc[idx]['text_embedding'])
            video_embedding = np.array(self.data.iloc[idx]['video_embedding'])

        video_tensor = load_video(path, num_frames=self.num_frames)


        # Convert label to integer using label_encoder
        label_idx = self.label_encoder.transform([label])[0]  # Convert string label to integer index

        return video_tensor, caption, label_idx, text_embedding, video_embedding  # Return the label index as an integer tensor


if __name__ == '__main__':
    # Run the function
    home_path = '/Users/user/PycharmProjects/frozen-in-time'
    home_path = '/mnt/iusers01/mace01/t08341gt/UCF_cap_mh'
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