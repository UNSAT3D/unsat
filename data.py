"""
Classes for creating torch dataloaders
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, random_split


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
        self.hdf5_file = h5py.File(hdf5_path, 'r')
        self.selection = data_selection

    def __len__(self):
        return self.selection.num_points

    def __getitem__(self, idx):
        (sample_idx, data_idx) = divmod(idx, self.selection.points_per_sample)

        (day_idx, height_idx) = divmod(data_idx, self.selection.num_heights)
        day_idx += self.selection.day_range[0]
        height_idx += self.selection.height_range[0]

        data = self.hdf5_file[self.selection.sample_list[sample_idx]]['data']
        data = data[day_idx, height_idx]
        labels = self.hdf5_file[self.selection.sample_list[sample_idx]]['labels'][
            day_idx, height_idx
        ]

        data = torch.from_numpy(data)
        labels = torch.from_numpy(labels)

        return data, labels


def create_dataloaders(
    hdf5_path: str,
    train_samples: List[str],
    height_range: Tuple[int, int],
    train_day_range: Tuple[int, int],
    validation_split: float,
    seed: int = 42,
    batch_size: int = 1,
) -> Dict[str, DataSelection]:
    """
    Create train/validation/test split of data.

    Selections of samples and days are made for the training/validation sets, the remaining are
    used for the test set.
    A selection of heights is made across all datasets.

    The test set is split into two parts:
    - test_strict: no overlap in samples or days with training/validation sets
    - test_overlap: overlap in either samples or days with training/validation sets

    Args:
        hdf5_path: path to hdf5 file containing all data
        train_samples: list of sample names to use for training
        height_range: tuple of (min, max) height indices to use
        train_day_range: tuple of (min, max) day indices to use for training
        validation_split: fraction of training data to use for validation
        seed: random seed for splitting data
        batch_size: batch size for dataloaders

    Returns:
        dataloaders: dict of dataloaders
            The keys are 'train', 'val', 'test_strict', 'test_overlap'
    """
    datasets = {}
    train_val_selection = DataSelection(
        sample_list=train_samples, height_range=height_range, day_range=train_day_range
    )
    train_val_dataset = XRayDataset(
        hdf5_path=hdf5_path, data_selection=train_val_selection, name='train_val'
    )

    # split train/val randomly
    num_val_samples = int(validation_split * len(train_val_dataset))
    num_train_samples = len(train_val_dataset) - num_val_samples
    datasets['train'], datasets['val'] = random_split(
        train_val_dataset, [num_train_samples, num_val_samples]
    )
    datasets['train'].name = 'train'
    datasets['val'].name = 'val'

    # find test set by removing train/val samples
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        all_samples = get_all_group_paths(hdf5_file)
        test_samples = list(set(all_samples) - set(train_samples))

        total_days = hdf5_file[all_samples[0]]['data'].shape[0]
        # Note: assumes training days start from 0
        test_day_range = (train_day_range[1], total_days)

    # The test set that has no overlaps in either samples or days
    strict_test_selection = DataSelection(
        sample_list=test_samples, height_range=height_range, day_range=test_day_range
    )
    datasets['test_strict'] = XRayDataset(
        hdf5_path=hdf5_path, data_selection=strict_test_selection, name='test_strict'
    )

    # The test set that has overlaps in either samples or days
    overlap_test_selection_same_days = DataSelection(
        sample_list=test_samples, height_range=height_range, day_range=train_day_range
    )
    overlap_test_dataset_same_days = XRayDataset(
        hdf5_path=hdf5_path,
        data_selection=overlap_test_selection_same_days,
        name='test_overlap_same_days',
    )
    overlap_test_selection_same_samples = DataSelection(
        sample_list=train_samples, height_range=height_range, day_range=test_day_range
    )
    overlap_test_dataset_same_samples = XRayDataset(
        hdf5_path=hdf5_path,
        data_selection=overlap_test_selection_same_samples,
        name='test_overlap_same_samples',
    )

    datasets['test_overlap'] = ConcatDataset(
        [overlap_test_dataset_same_days, overlap_test_dataset_same_samples]
    )
    datasets['test_overlap'].name = 'test_overlap'

    # turn into dataloaders
    dataloaders = {}
    for name, dataset in datasets.items():
        dataloaders[name] = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=(name == 'train')
        )

    return dataloaders


def get_all_group_paths(hdf5_file):
    leaf_group_paths = []

    def check_leaf_group(name):
        item = hdf5_file[name]
        if isinstance(item, h5py.Group):
            # Check if it has datasets as children
            if any(isinstance(item[obj_name], h5py.Dataset) for obj_name in item):
                leaf_group_paths.append(name)

    hdf5_file.visit(check_leaf_group)
    return leaf_group_paths
