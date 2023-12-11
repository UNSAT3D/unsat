"""
Classes for creating torch dataloaders
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class DataSelection:
    """
    DataSelection object expressing which subset of data to use.

    Args:
        sample_list: list of sample names to use
        height_range: tuple of (min, max) height indices to use
        day_range: tuple of (min, max) day indices to use
    """

    sample_list: List[str]
    height_range: Tuple[int, int]
    day_range: Tuple[int, int]

    @property
    def num_samples(self):
        return len(self.sample_list)

    @property
    def num_heights(self):
        return self.height_range[1] - self.height_range[0]

    @property
    def num_days(self):
        return self.day_range[1] - self.day_range[0]

    @property
    def points_per_sample(self):
        return self.num_heights * self.num_days

    @property
    def num_points(self):
        return self.num_samples * self.points_per_sample


class XRayDataset(Dataset):
    """
    Dataset representing a selection of the total data (train/validation/test split)

    Args:
        hdf5_path: path to hdf5 file containing all data
        data_selection: DataSelection object expressing which subset of data to use
        name: name of dataset (train/validation/test)
    """

    def __init__(self, hdf5_path: str, data_selection: DataSelection, name: str):
        self.name = name
        self.hdf5_path = hdf5_path
        self.selection = data_selection

    def __len__(self):
        return self.selection.num_points

    def __getitem__(self, idx):
        (sample_idx, data_idx) = divmod(idx, self.selection.points_per_sample)

        (day_idx, height_idx) = divmod(data_idx, self.selection.num_heights)
        day_idx += self.selection.day_range[0]
        height_idx += self.selection.height_range[0]

        with h5py.File(self.hdf5_path, 'r') as hdf5_file:
            data = hdf5_file[self.selection.sample_list[sample_idx]]['data']
            data = data[day_idx, height_idx]
            labels = hdf5_file[self.selection.sample_list[sample_idx]]['labels'][
                day_idx, height_idx
            ]

            data = torch.from_numpy(data)
            labels = torch.from_numpy(labels)

            return data, labels
