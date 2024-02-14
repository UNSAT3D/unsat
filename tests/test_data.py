import itertools
import os

import h5py
import numpy as np
import pytest

from unsat.data import DataSelection, XRayDataModule, XRayDataset

DAYS = 4
HEIGHT = 20
WIDTH = 3
DEPTH = 3
CHANNELS = 1

BATCH_SIZE = 4
VAL_SPLIT = 0.4
(HEIGHT_START, HEIGHT_END) = (5, 10)
(DAY_START, DAY_END) = (0, 2)

SAMPLES = ['coarse/loose/04', 'fine/dense/07']


@pytest.fixture
def create_test_h5():
    np.random.seed(0)
    hdf5_path = 'test_data.h5'
    shape = (DAYS, HEIGHT, WIDTH, DEPTH)
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
    selection = DataSelection(
        sample_list=SAMPLES, height_range=(HEIGHT_START, HEIGHT_END), day_range=(DAY_START, DAY_END)
    )
    dataset = XRayDataset(hdf5_path=create_test_h5, data_selection=selection, name='test')

    # test shapes
    assert len(dataset) == len(SAMPLES) * (HEIGHT_END - HEIGHT_START) * (DAY_END - DAY_START)
    data_sample, label_sample = dataset[-1]
    assert data_sample.shape == (CHANNELS, WIDTH, DEPTH)
    assert label_sample.shape == (WIDTH, DEPTH)

    # test some values
    with h5py.File(create_test_h5, 'r') as hdf5_file:
        # First sample, first day
        sample_from_dataset = dataset[0][0].numpy()[0]  # we added a channel dimension
        sample_from_hdf5 = hdf5_file['coarse/loose/04']['data'][0, 5]
        np.testing.assert_allclose(sample_from_dataset, sample_from_hdf5, atol=1e-6)

        # First sample, second day
        num_heights = HEIGHT_END - HEIGHT_START
        sample_from_dataset = dataset[num_heights][0].numpy()[0]
        sample_from_hdf5 = hdf5_file['coarse/loose/04']['data'][1, 5]
        np.testing.assert_allclose(sample_from_dataset, sample_from_hdf5, atol=1e-6)

        # second sample, first day
        num_days = DAY_END - DAY_START
        sample_from_dataset = dataset[num_days * num_heights][0].numpy()[0]
        sample_from_hdf5 = hdf5_file['fine/dense/07']['data'][0, 5]
        np.testing.assert_allclose(sample_from_dataset, sample_from_hdf5, atol=1e-6)


def test_dataloaders(create_test_h5, remove_test_h5):
    sample_list = ['coarse/loose/04']
    data_module = XRayDataModule(
        hdf5_path=create_test_h5,
        train_samples=sample_list,
        height_range=(HEIGHT_START, HEIGHT_END),
        train_day_range=(DAY_START, DAY_END),
        validation_split=VAL_SPLIT,
        batch_size=BATCH_SIZE,
        seed=0,
        num_workers=1,
    )
    data_module.prepare_data()
    dataloaders = data_module.dataloaders

    # test shapes
    for name, loader in dataloaders.items():
        x, y = next(iter(loader))
        assert x.shape == (BATCH_SIZE, CHANNELS, WIDTH, DEPTH)
        assert y.shape == (BATCH_SIZE, WIDTH, DEPTH)

    # test counts
    train_test_count = len(sample_list) * (HEIGHT_END - HEIGHT_START) * (DAY_END - DAY_START)
    assert len(dataloaders['val'].dataset) + len(dataloaders['train'].dataset) == train_test_count

    total_count = len(SAMPLES) * (HEIGHT_END - HEIGHT_START) * DAYS
    test_count = total_count - train_test_count
    assert (
        len(dataloaders['test_strict'].dataset) + len(dataloaders['test_overlap'].dataset)
        == test_count
    )

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
