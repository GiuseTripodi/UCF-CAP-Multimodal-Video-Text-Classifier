from os.path import join
import pandas as pd
import re
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.plotting_functions import analyze_dataset_statistics, plot_class_distribution, plot_samples_per_class, \
    plot_split_distribution, plot_caption_length_distribution, plot_class_distribution_per_split


def create_csv_splits(home_path):
    # Paths to the UCF101 dataset
    data_dir = f'{home_path}/data/UcfCap/YouTubeClips'
    output_dir = f'{home_path}/data/UcfCap/'
    mapping_path = f'{home_path}/data/UcfCap/captions/youtube_mapping.txt'

    # Load mapping
    mapping = {}
    with open(mapping_path, "r") as file:
        for line in file:
            key, value = line.strip().split()
            mapping[key] = value

    # Get the list of class names
    videos_folder_frame = sorted(os.listdir(data_dir))

    # Collect video paths and their labels
    data_entries = []
    for video_fold in videos_folder_frame:
        video_path = os.path.join(data_dir, video_fold)
        match = re.search(r'([a-zA-Z]+)\d', video_fold)
        if match and os.path.isdir(video_path):
            label = match.group(1)
            data_entries.append((video_path, label, mapping[video_fold]))

    df_mapping = pd.DataFrame(data_entries, columns=["video_path", "label", "videoID"])

    def load_txt_splitting(path):
        with open(path, "r", encoding="utf-8") as file:
            data = [line.strip().split(" ", 1) for line in file]
        df = pd.DataFrame(data, columns=["videoID", "caption"])
        return df

    # Load test, train, val txt file and merge
    df_test = load_txt_splitting(f'{home_path}/data/UcfCap/captions/sents_test_lc_nopunc.txt')
    df_val = load_txt_splitting(f'{home_path}/data/UcfCap/captions/sents_val_lc_nopunc.txt')
    df_train = load_txt_splitting(f'{home_path}/data/UcfCap/captions/sents_train_lc_nopunc.txt')

    # Merge datasets
    merged_df_test = pd.merge(df_test, df_mapping, on="videoID", how="inner")
    merged_df_val = pd.merge(df_val, df_mapping, on="videoID", how="inner")
    merged_df_train = pd.merge(df_train, df_mapping, on="videoID", how="inner")

    # Save datasets
    merged_df_test.to_csv(join(output_dir, "test_dataset.csv"), index=False)
    merged_df_val.to_csv(join(output_dir, "val_dataset.csv"), index=False)
    merged_df_train.to_csv(join(output_dir, "train_dataset.csv"), index=False)

    return merged_df_train, merged_df_val, merged_df_test




if __name__ == '__main__':
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'

    # Run the function
    home_path = '/Users/user/PycharmProjects/frozen-in-time'
    home_path = '/mnt/iusers01/mace01/t08341gt/UCF_cap_mh'
    output_dir = f'{home_path}/data/UcfCap/'

    print("Creating CSV splits...")
    df_train, df_val, df_test = create_csv_splits(home_path)

    print("\nGenerating dataset analysis and visualizations...")

    # Generate all analyses
    analyze_dataset_statistics(df_train, df_val, df_test, output_dir)
    plot_class_distribution(df_train, df_val, df_test, output_dir)
    plot_samples_per_class(df_train, df_val, df_test, output_dir, top_n=20)
    plot_split_distribution(df_train, df_val, df_test, output_dir)
    plot_caption_length_distribution(df_train, df_val, df_test, output_dir)
    plot_class_distribution_per_split(df_train, df_val, df_test, output_dir)

    print("\n" + "=" * 60)
    print("Analysis complete! All visualizations saved to:", output_dir)
    print("=" * 60)
