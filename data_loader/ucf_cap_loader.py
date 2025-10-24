import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import glob
import os


def load_video_interpolate(path, num_frames=8, img_size=224):
    """
    Load video with temporal interpolation to fixed frame count.
    Better for transformers as it preserves temporal continuity.

    Args:
        path: Directory containing video frames as images
        num_frames: Target number of frames
        img_size: Target spatial size (height, width)

    Returns:
        video tensor of shape (num_frames, 3, img_size, img_size)
    """
    # Load all frames
    frame_paths = sorted(glob.glob(os.path.join(path, '*.jpg')))

    if len(frame_paths) == 0:
        raise ValueError(f"No frames found in {path}")

    # Transforms for loading
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load and stack frames
    frames = []
    for frame_path in frame_paths:
        img = Image.open(frame_path).convert("RGB")
        frame = transform(img)  # (3, H, W)
        frames.append(frame)

    video = torch.stack(frames)  # (T, 3, H, W)

    # Temporal interpolation if needed
    if len(frames) != num_frames:
        # Rearrange to (1, C, T, H, W) for interpolation
        video = video.permute(1, 0, 2, 3).unsqueeze(0)  # (1, 3, T, H, W)

        # Interpolate temporal dimension
        video = F.interpolate(
            video,
            size=(num_frames, img_size, img_size),
            mode='trilinear',
            align_corners=False
        )

        # Back to (T, C, H, W)
        video = video.squeeze(0).permute(1, 0, 2, 3)  # (T, 3, H, W)

    return video


def load_video_uniform_sample(path, num_frames=8, img_size=224):
    """
    Load video with uniform temporal sampling.
    Faster but may lose temporal smoothness.

    Args:
        path: Directory containing video frames
        num_frames: Target number of frames
        img_size: Target spatial size

    Returns:
        video tensor of shape (num_frames, 3, img_size, img_size)
    """
    frame_paths = sorted(glob.glob(os.path.join(path, '*.jpg')))

    if len(frame_paths) == 0:
        raise ValueError(f"No frames found in {path}")

    total_frames = len(frame_paths)

    # Sample frame indices uniformly
    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        # Repeat frames if video is too short
        indices = np.array([i % total_frames for i in range(num_frames)])

    # Load selected frames
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    frames = []
    for idx in indices:
        img = Image.open(frame_paths[idx]).convert("RGB")
        frame = transform(img)
        frames.append(frame)

    video = torch.stack(frames)  # (T, 3, H, W)
    return video


class UCF101Dataset(Dataset):
    """
    Video dataset for action recognition.
    """

    def __init__(self, csv_file, num_frames=8, img_size=224,
                 sampling_method='interpolate', transform=None):
        """
        Args:
            csv_file: Path to CSV/pickle with columns: videoID, caption, video_path, label
            num_frames: Number of frames to extract
            img_size: Spatial resolution (assumes square)
            sampling_method: 'interpolate' or 'uniform'
            transform: Additional transforms (optional, applied after loading)
        """
        self.num_frames = num_frames
        self.img_size = img_size
        self.sampling_method = sampling_method
        self.transform = transform

        # Load data
        if csv_file.endswith('.csv'):
            df = pd.read_csv(csv_file)
        else:
            df = pd.read_pickle(csv_file)

        self.data = shuffle(df, random_state=42)

        # Encode labels
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.data['label'])

        print(f"\nDataset loaded: {len(self.data)} videos")
        print(f"Sampling method: {sampling_method}")
        print(f"Target frames: {num_frames}, Image size: {img_size}x{img_size}")
        print(f"Classes: {list(self.label_encoder.classes_)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        video_id = row['videoID']
        caption = row['caption']
        path = row['video_path']
        label = row['label']

        try:
            # Load video based on sampling method
            if self.sampling_method == 'interpolate':
                video = load_video_interpolate(path, self.num_frames, self.img_size)
            elif self.sampling_method == 'uniform':
                video = load_video_uniform_sample(path, self.num_frames, self.img_size)
            else:
                raise ValueError(f"Unknown sampling method: {self.sampling_method}")

            # Additional transforms if provided
            if self.transform:
                video = self.transform(video)

            # Encode label
            label_idx = self.label_encoder.transform([label])[0]

            return video, caption, label_idx

        except Exception as e:
            print(f"Error loading video {video_id} at {path}: {e}")
            # Return next video
            return self.__getitem__((idx + 1) % len(self))


def collate_video_batch(batch):
    """
    Custom collate function for video batches.
    Handles videos, captions, and labels.
    """
    videos = torch.stack([item[0] for item in batch])  # (B, T, C, H, W)
    captions = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)

    return videos, captions, labels


# ============ USAGE EXAMPLES ============

if __name__ == '__main__':
    csv_file = '/Users/user/PycharmProjects/frozen-in-time/data/UcfCap/val_dataset.csv'

    # Example 1: Interpolation (recommended for transformers)
    print("=" * 60)
    print("Testing with INTERPOLATION")
    print("=" * 60)
    dataset_interp = UCF101Dataset(
        csv_file,
        num_frames=16,
        img_size=224,
        sampling_method='interpolate'
    )

    dataloader_interp = DataLoader(
        dataset_interp,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_video_batch,
        num_workers=2
    )

    # Test batch
    videos, captions, labels = next(iter(dataloader_interp))
    print(f"\nBatch shapes:")
    print(f"  Videos: {videos.shape}")  # (4, 16, 3, 224, 224)
    print(f"  Captions: {len(captions)} strings")
    print(f"  Labels: {labels.shape}")  # (4,)

    # Example 2: Uniform sampling (faster, simpler)
    print("\n" + "=" * 60)
    print("Testing with UNIFORM SAMPLING")
    print("=" * 60)
    dataset_uniform = UCF101Dataset(
        csv_file,
        num_frames=16,
        img_size=224,
        sampling_method='uniform'
    )

    dataloader_uniform = DataLoader(
        dataset_uniform,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_video_batch
    )

    videos, captions, labels = next(iter(dataloader_uniform))
    print(f"\nBatch shapes:")
    print(f"  Videos: {videos.shape}")
    print(f"  Captions: {len(captions)} strings")
    print(f"  Labels: {labels.shape}")

    # Example 3: For SpaceTimeTransformer input format
    print("\n" + "=" * 60)
    print("SpaceTimeTransformer Input Format")
    print("=" * 60)

    # SpaceTime expects (B, C, T, H, W)
    videos_st = videos.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W) -> (B, C, T, H, W)
    print(f"SpaceTime format: {videos_st.shape}")  # (4, 3, 16, 224, 224)