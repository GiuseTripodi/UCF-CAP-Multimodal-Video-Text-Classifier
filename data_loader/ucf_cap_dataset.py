import pandas as pd
import os
import csv
import re
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

def create_csv_splits(home_path):
    # Paths to the UCF101 dataset
    data_dir = f'{home_path}/data/UcfCap/YouTubeClips'  # Replace with your UCF101 frames directory
    output_dir = f'{home_path}/data/UcfCap/'  # Directory to save CSV files

    # Get the list of class names (folder names in the dataset)
    videos_folder_frame = sorted(os.listdir(data_dir))

    # Collect video paths and their labels
    data_entries = []
    for video_fold in videos_folder_frame:
        video_path = os.path.join(data_dir, video_fold)
        match = re.search(r'([a-zA-Z]+)\d', video_fold)
        if match and os.path.isdir(video_path):  # Ensure it's a directory and matches the pattern
            label = match.group(1)
            data_entries.append((video_path, label))

    # Split data into train, test, and validation sets
    train_entries, temp_entries = train_test_split(data_entries, test_size=0.4, random_state=42)  # 60% train, 40% temp
    val_entries, test_entries = train_test_split(temp_entries, test_size=0.5, random_state=42)  # 20% val, 20% test

    # Helper function to write CSV
    def write_csv(entries, filename):
        csv_path = os.path.join(output_dir, filename)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['video_path', 'label'])  # Add a header row
            writer.writerows(entries)
        print(f"CSV file created: {csv_path}")

    # Write to separate CSV files
    #TODO test with less data
    write_csv(train_entries, 'train_dataset.csv')
    write_csv(val_entries, 'val_dataset.csv')
    write_csv(test_entries, 'test_dataset.csv')

class UCF101Dataset(Dataset):
    def __init__(self, csv_file, transform=None, num_frames=8):
        self.data = []
        self.num_frames = num_frames
        self.transform = transform

        # Initialize a LabelEncoder to convert string labels to integer indices
        self.label_encoder = LabelEncoder()

        # Read and process the CSV file
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header
            for line in reader:
                path, label = line
                self.data.append((path, label))


        # Fit the label encoder to the labels
        self.label_encoder.fit([label for _, label in self.data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        frames = sorted(glob.glob(os.path.join(path, '*.jpg')))
        if len(frames) < self.num_frames:
            frames = frames + frames[:self.num_frames - len(frames)]  # Padding

        selected_frames = frames[:self.num_frames]  # Choose first N frames
        images = [Image.open(frame) for frame in selected_frames]

        if self.transform:
            images = [self.transform(img) for img in images]

        video_tensor = torch.stack(images, dim=0)  # Shape: [num_frames, C, H, W]

        # Convert label to integer using label_encoder
        label_idx = self.label_encoder.transform([label])[0]  # Convert string label to integer index

        return video_tensor, label_idx  # Return the label index as an integer tensor


if __name__ == '__main__':
    # Run the function
    #home_path_local = '/Users/user/PycharmProjects/frozen-in-time'
    home_path_clus = '/mnt/iusers01/mace01/t08341gt/UCF_cap_mh'
    create_csv_splits(home_path_clus)

    '''

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    csv_file = '/Users/user/PycharmProjects/frozen-in-time/data/UcfCap/dataset.csv'

    dataset = UCF101Dataset(csv_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    '''