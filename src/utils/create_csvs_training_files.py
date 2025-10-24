from os.path import join
import pandas as pd
import re
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



if __name__ == '__main__':
    # Run the function
    home_path = '/Users/user/PycharmProjects/frozen-in-time'
    home_path = '/mnt/iusers01/mace01/t08341gt/UCF_cap_mh'
    create_csv_splits(home_path)