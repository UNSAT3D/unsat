import itertools
import os

import h5py
import numpy as np
import pytest

from unsatIO.data import DataSelection, XRayDataset, create_dataloaders

DAYS = 4
HEIGHT = 20
WIDTH = 3
DEPTH = 3
CHANNELS = 1
SAMPLES = ['coarse/loose/04', 'fine/dense/07']


@pytest.fixture
def create_test_h5():
    np.random.seed(0)
    hdf5_path = 'test_data.h5'
    shape = (DAYS, HEIGHT, WIDTH, DEPTH, CHANNELS)
    with h5py.File(hdf5_path, 'w') as hdf5_file:
        for sample in SAMPLES:
            hdf5_file.create_dataset(f'{sample}/data', data=np.random.rand(*shape))
            hdf5_file.create_dataset(f'{sample}/labels', data=np.random.rand(*shape))
    return hdf5_path


@pytest.fixture
def remove_test_h5(create_test_h5):
    yield
    os.remove(create_test_h5)


def test_dataset(create_test_h5, remove_test_h5):
    (height_start, height_end) = (5, 10)
    (day_start, day_end) = (0, 2)
    selection = DataSelection(
        sample_list=SAMPLES, height_range=(height_start, height_end), day_range=(day_start, day_end)
    )
    dataset = XRayDataset(hdf5_path=create_test_h5, data_selection=selection, name='test')

    # test shapes
    assert len(dataset) == len(SAMPLES) * (height_end - height_start) * (day_end - day_start)
    assert dataset[-1][0].shape == (WIDTH, DEPTH, CHANNELS)
    assert dataset[-1][1].shape == (WIDTH, DEPTH, CHANNELS)

    # test some values
    with h5py.File(create_test_h5, 'r') as hdf5_file:
        np.testing.assert_equal(
            dataset[0][0].numpy(), hdf5_file['coarse/loose/04']['data'][0, 5], atol=1e-6
        )
        # second day
        np.testing.assert_equal(
            dataset[5][0].numpy(), hdf5_file['coarse/loose/04']['data'][1, 5], atol=1e-6
        )
        # second sample
        np.testing.assert_equal(
            dataset[10][0].numpy(), hdf5_file['fine/dense/07']['data'][0, 5], atol=1e-6
        )


def test_dataloaders(create_test_h5, remove_test_h5):
    (height_start, height_end) = (5, 10)
    (day_start, day_end) = (0, 3)
    sample_list = ['coarse/loose/04']
    dataloaders = create_dataloaders(
        hdf5_path=create_test_h5,
        train_samples=sample_list,
        height_range=(height_start, height_end),
        train_day_range=(day_start, day_end),
        validation_split=0.2,
        batch_size=1,
    )

    # test shapes
    for name, loader in dataloaders.items():
        assert loader.dataset[0][0].shape == (WIDTH, DEPTH, CHANNELS)
        assert loader.dataset[0][1].shape == (WIDTH, DEPTH, CHANNELS)

    # test counts
    train_test_count = len(sample_list) * (height_end - height_start) * (day_end - day_start)
    assert len(dataloaders['val']) + len(dataloaders['train']) == train_test_count

    total_count = len(SAMPLES) * (height_end - height_start) * DAYS
    test_count = total_count - train_test_count
    assert len(dataloaders['test_strict']) + len(dataloaders['test_overlap']) == test_count

    # test overlaps
    elements = {}
    for name, loader in dataloaders.items():
        elements[name] = set()
        for data, _ in loader:
            elements[name].add(data.numpy().tobytes())

    for set1, set2 in itertools.product(dataloaders.keys(), repeat=2):
        if set1 == set2:
            continue
        assert len(elements[set1].intersection(elements[set2])) == 0
