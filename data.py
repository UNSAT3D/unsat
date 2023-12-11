"""
Classes for creating torch dataloaders
"""
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class XRayDataset(Dataset):
    def __init__(
        self,
        hdf5_path: str,
        height_range: Tuple[int, int],
        day_range: Tuple[int, int],
        sample_list: List[str],
    ):
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
            data = hdf5_file[self.sample_list[sample_idx]]['data'][day_idx, height_idx]
            labels = hdf5_file[self.sample_list[sample_idx]]['labels'][day_idx, height_idx]

            data = torch.from_numpy(data)
            labels = torch.from_numpy(labels)

            return data, labels


def main():
    tst = XRayDataset(
        '../data/data.h5', (1_000, 1_100), (2, 5), ['coarse/loose/04', 'fine/dense/07']
    )

    print(len(tst))
    print(tst[0][0].shape, tst[0][1].shape)


if __name__ == '__main__':
    main()
