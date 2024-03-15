import h5py
from lightning.pytorch.callbacks import Callback
import numpy as np
import torch

import wandb


class ClassWeightsCallback(Callback):
    """
    Before starting a fit, compute the relative frequencies of the classes in the training data,
    and use the inverse (normalized) as class weights for the loss function.

    This prevents rare classes from being ignored by the loss function.
    """

    def on_fit_start(self, trainer, pl_module):
        train_dataloader = trainer.datamodule.dataloaders['train']
        class_counts = torch.zeros(pl_module.num_classes)
        for _, labels in train_dataloader:
            # labels is a high rank tensor with integer entries between 1 and pl_module.num_classes
            counts = torch.bincount(labels.flatten(), minlength=pl_module.num_classes)
            class_counts += counts

        class_freqs = class_counts / class_counts.sum()
        class_weights = 1 / class_freqs

        # In case the training set doesn't include all classes, we need to set the weights for
        # the missing classes to zero.
        class_weights[class_counts == 0] = 0.0

        # Need to send the class weights to the device to avoid errors
        pl_module.class_weights = class_weights.to(pl_module.device)


class CheckFaultsCallback(Callback):
    """
    After every training epoch, compute the model predictions on a set of examples where the
    labels are known to be wrong, and upload plots of these to wandb.
    """

    def __init__(self, faults_path: str):
        self.faults_path = faults_path

        # for each example, store this info
        self.samples = []
        self.centers = []
        self.issues = []
        self.data = []
        self.labels = []

        with h5py.File(faults_path, 'r') as f:

            def store_faults(name, obj):
                if isinstance(obj, h5py.Group) and 'data' in obj and 'labels' in obj:
                    sample, center = name.rsplit('/', 1)
                    self.samples.append(sample)
                    self.centers.append(center)
                    self.issues.append(obj.attrs['issue'])
                    self.data.append(obj['data'][()])
                    self.labels.append(obj['labels'][()])

            f.visititems(store_faults)

        self.data = torch.from_numpy(np.stack(self.data)).type(torch.float32)
        self.data = self.data.unsqueeze(1)
        self.labels = torch.from_numpy(np.stack(self.labels)).type(torch.long)

    def on_train_epoch_end(self, trainer, pl_module):
        preds = pl_module.network(self.data.to(pl_module.device)).argmax(dim=1).to('cpu')

        class_labels = {i: name for i, name in enumerate(pl_module.class_names)}
        for i, (sample, issue, center) in enumerate(zip(self.samples, self.issues, self.centers)):
            plot = wandb.Image(
                self.data.squeeze(1)[i].numpy(),
                caption=f"Sample: {sample}, Issue: {issue} Center: {center}",
                masks={
                    "predictions": {"mask_data": preds[i].numpy(), "class_labels": class_labels},
                    "labels": {"mask_data": self.labels[i].numpy(), "class_labels": class_labels},
                },
            )

            wandb.log({f"faults/{issue}/{sample}/{center}": plot})
