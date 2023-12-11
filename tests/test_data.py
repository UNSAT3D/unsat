import os

import h5py
import numpy as np
import pytest

from data import DataSelection, XRayDataset

DAYS = 4
HEIGHT = 20
WIDTH = 3
DEPTH = 3
CHANNELS = 1


@pytest.fixture
def create_test_h5():
    hdf5_path = 'test_data.h5'
    samples = ['coarse/loose/04', 'fine/dense/07']
    shape = (DAYS, HEIGHT, WIDTH, DEPTH, CHANNELS)
    with h5py.File(hdf5_path, 'w') as hdf5_file:
        for sample in samples:
            hdf5_file.create_dataset(f'{sample}/data', data=np.random.rand(*shape))
            hdf5_file.create_dataset(f'{sample}/labels', data=np.random.rand(*shape))
    return hdf5_path


@pytest.fixture
def remove_test_h5(create_test_h5):
    yield
    os.remove(create_test_h5)


def test_dataset(create_test_h5, remove_test_h5):
    (height_start, height_end) = (5, 10)
    (day_start, day_end) = (1, 3)
    sample_list = ['coarse/loose/04', 'fine/dense/07']
    selection = DataSelection(
        sample_list=sample_list,
        height_range=(height_start, height_end),
        day_range=(day_start, day_end),
    )
    dataset = XRayDataset(hdf5_path=create_test_h5, data_selection=selection, name='test')

    # test shapes
    assert len(dataset) == len(sample_list) * (height_end - height_start) * (day_end - day_start)
    assert dataset[-1][0].shape == (3, 3, 1)
    assert dataset[-1][1].shape == (3, 3, 1)

    # test some values
    with h5py.File(create_test_h5, 'r') as hdf5_file:
        print(dataset[0][0].numpy() - hdf5_file['coarse/loose/04']['data'][1, 5])
        np.testing.assert_equal(dataset[0][0].numpy(), hdf5_file['coarse/loose/04']['data'][1, 5])
        # second day
        np.testing.assert_equal(dataset[5][0].numpy(), hdf5_file['coarse/loose/04']['data'][2, 5])
        # second sample
        np.testing.assert_equal(dataset[10][0].numpy(), hdf5_file['fine/dense/07']['data'][1, 5])
