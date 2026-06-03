import numpy as np
import torch

from data_loader.ucf_cap_loader import UCF101Dataset, collate_video_batch
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader

from src.utils.parse_config import ConfigParser

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_dataset(config: ConfigParser):
    """Load UCF-CAP frames and associated text descriptions."""
    csv_train_file_path, csv_val_file_path = config.train_path

    dataset_train = UCF101Dataset(
        csv_train_file_path,
        sampling_method=config.sampling_method,
        num_frames=config.num_frames,
        img_size=config.img_size
    )
    dataset_val = UCF101Dataset(
        csv_val_file_path,
        sampling_method=config.sampling_method,
        num_frames=config.num_frames,
        img_size=config.img_size
    )

    train_loader = DataLoader(
        dataset_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_video_batch
    )

    val_loader = DataLoader(
        dataset_val,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_video_batch
    )

    # Compute class weights using encoded labels
    train_labels = dataset_train.label_encoder.transform(dataset_train.data['label'])
    unique_labels = np.unique(train_labels)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_labels,
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    return train_loader, val_loader, class_weights
