import os

import h5py
import numpy as np
import pytest
import torch

from unsat.data import XRayDataModule
from unsat.train import _compute_loss

DAYS = 3
HEIGHT = 30
WIDTH = 20
DEPTH = 20
PATCH_SIZE = 8
PATCH_BORDER = 2
CHANNELS = 1
NUM_CLASSES = 5

BATCH_SIZE = 1
VAL_SPLIT = 0.4
(HEIGHT_START, HEIGHT_END) = (0, 20)
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
            hdf5_file.create_dataset(
                f'{sample}/labels', data=np.random.randint(0, NUM_CLASSES, size=shape)
            )
    return hdf5_path


@pytest.fixture
def remove_test_h5(create_test_h5):
    yield
    os.remove(create_test_h5)


def create_loader(create_test_h5, remove_test_h5, dimension):
    sample_list = ['coarse/loose/04']
    data_module = XRayDataModule(
        hdf5_path=create_test_h5,
        train_samples=sample_list,
        height_range=(HEIGHT_START, HEIGHT_END),
        train_day_range=(DAY_START, DAY_END),
        validation_split=VAL_SPLIT,
        batch_size=BATCH_SIZE,
        patch_size=PATCH_SIZE,
        patch_border=PATCH_BORDER,
        seed=0,
        num_workers=1,
        dimension=dimension,
        class_names=["water", "background", "air", "root", "soil"],
        input_channels=1,
    )
    data_module.prepare_data()
    train_loader = data_module.dataloaders['train']

    return train_loader


@pytest.mark.parametrize("dimension", [2, 3])
def test_mask(dimension, create_test_h5, remove_test_h5):
    """Test that the loss is independent of the masked border of a patch."""
    train_loader = create_loader(create_test_h5, remove_test_h5, dimension)

    for data, labels, mask in train_loader:
        preds = torch.rand((BATCH_SIZE, NUM_CLASSES, *labels.shape[1:]))

        loss_original = _compute_loss(preds, labels, mask, torch.ones(5))

        # Modify preds in the masked border
        preds_modified = preds + 100.0 * (1 - mask.float())

        loss_modified = _compute_loss(preds_modified, labels, mask, torch.ones(5))

        np.testing.assert_allclose(loss_original, loss_modified)

        break  # only test one batch
