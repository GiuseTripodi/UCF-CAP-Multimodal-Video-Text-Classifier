import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

# pick frames uniformly across the duration of the videos
def uniform_frame_sample(total_frames, num_samples=16):
    if total_frames < num_samples:
        # Loop or pad if too short
        indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    else:
        indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    return indices


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

        self.label_encoder = LabelEncoder()
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
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    csv_file = '/Users/user/PycharmProjects/frozen-in-time/data/UcfCap/val_dataset.csv'

    dataset = UCF101Dataset(csv_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
