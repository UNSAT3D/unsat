from lightning.pytorch.callbacks import Callback
import torch


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
