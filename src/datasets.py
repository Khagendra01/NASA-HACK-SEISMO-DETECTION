import os
import obspy
import torch
import pandas as pd
from torch.utils.data import Dataset

from .preprocessing import preprocess_signal


class SeismicDataset(Dataset):

    def __init__(
        self,
        file_paths,
        labels,
        lowcut,
        highcut,
        window_size_sec
    ):

        self.file_paths = file_paths
        self.labels = labels
        self.lowcut = lowcut
        self.highcut = highcut
        self.window_size_sec = window_size_sec

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):

        file_path = self.file_paths[idx]

        stream = obspy.read(file_path)

        data = stream[0].data
        fs = stream[0].stats.sampling_rate

        processed = preprocess_signal(
            data,
            fs,
            self.lowcut,
            self.highcut,
            self.window_size_sec
        )

        processed = torch.tensor(
            processed,
            dtype=torch.float32
        )

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.float32
        )

        return processed, label