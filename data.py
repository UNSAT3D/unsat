"""
Classes for creating torch dataloaders
"""
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class XRayDataset(Dataset):
    """
    Dataset representing a selection of the total data (train/validation/test split)

    Args:
        hdf5_path: path to hdf5 file containing all data
        height_range: range of height indices to include in dataset
        day_range: range of day indices to include in dataset
        sample_list: list of samples to include in dataset
        name: name of dataset (train/validation/test)
    """

    def __init__(
        self,
        hdf5_path: str,
        height_range: Tuple[int, int],
        day_range: Tuple[int, int],
        sample_list: List[str],
        name: str,
    ):
        self.name = name
        self.hdf5_path = hdf5_path

        self.sample_list = sample_list
        self.height_range = height_range
        self.day_range = day_range

        self.heights = height_range[1] - height_range[0]
        self.days = day_range[1] - day_range[0]
        self.points_per_sample = self.heights * self.days

    def __len__(self):
        return self.points_per_sample * len(self.sample_list)

    def __getitem__(self, idx):
        sample_idx = idx // self.points_per_sample
        data_idx = idx % self.points_per_sample

        day_idx = self.day_range[0] + (data_idx // self.heights)
        height_idx = self.height_range[0] + (data_idx % self.heights)

        with h5py.File(self.hdf5_path, 'r') as hdf5_file:
            data = hdf5_file[self.sample_list[sample_idx]]['data']
            data = data[day_idx, height_idx]
            labels = hdf5_file[self.sample_list[sample_idx]]['labels'][day_idx, height_idx]

            data = torch.from_numpy(data)
            labels = torch.from_numpy(labels)

            return data, labels
