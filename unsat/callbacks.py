from lightning.pytorch.callbacks import Callback
import torch
import wandb


class ClassWeightsCallback(Callback):
    """
    Before starting a fit, compute the relative frequencies of the classes in the training data,
    and use the inverse (normalized) as class weights for the loss function.

    This prevents rare classes from being ignored by the loss function.
    """

    def on_fit_start(self, trainer, pl_module):
        train_dataloader = trainer.datamodule.dataloaders["train"]
        class_counts = torch.zeros(pl_module.num_classes)
        for _, labels, _ in train_dataloader:
            # labels is a high rank tensor with integer entries between 1 and pl_module.num_classes
            counts = torch.bincount(labels.flatten(), minlength=pl_module.num_classes)
            class_counts += counts

        class_freqs = class_counts / class_counts.sum()
        class_weights = 1 / class_freqs

        # In case the training set doesn't include all classes, we need to set the weights for
        # the missing classes to zero.
        class_weights[class_counts == 0] = 0.0

        class_weights /= class_weights.sum()

        # Need to send the class weights to the device to avoid errors
        pl_module.class_weights = class_weights.to(pl_module.device)


class CheckFaultsCallback(Callback):
    """
    After every training epoch, compute the model predictions on a set of examples where the
    labels are known to be wrong, and upload plots of these to wandb.
    """

    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def on_fit_start(self, trainer, pl_module):
        self.faults_dataloader = trainer.datamodule.dataloaders["faults"]
        self.dimension = pl_module.network.dimension

        self.samples = []
        self.days = []
        self.issues = []
        self.centers = []
        self.splits = []
        self.labels = []
        self.data = []

        for data, labels, sample, day, center, issue, split in self.faults_dataloader:
            self.samples.extend(sample)
            self.days.extend(day)
            self.issues.extend(issue)
            self.splits.extend(split)

            self.centers.append(center)
            self.labels.append(labels)
            self.data.append(data)

        self.centers = torch.cat(self.centers, dim=0).to("cpu")
        self.labels = torch.cat(self.labels, dim=0).to("cpu")
        self.data = torch.cat(self.data, dim=0).to("cpu")

        self.data = self.extract_patch(self.data)
        self.labels = self.extract_patch(self.labels)

    def extract_patch(self, data):
        """Extract a 2D region of size patch_size around the center."""
        # NOTE: this assumes that these are already patches with the center in the middle
        # this will fail to be focussed on the point of interest if either no patch is used,
        # or the center point is close to the edge of the full image

        spatial_shape = torch.tensor(data.shape[1:])

        # for 3D inputs, extract middle slice in the horizontal direction
        if self.dimension == 3:
            data = data[:, spatial_shape[0] // 2]
            spatial_shape = spatial_shape[1:]

        start = spatial_shape // 2 - self.patch_size // 2
        end = spatial_shape // 2 + self.patch_size // 2
        slices = [slice(s, e) for s, e in zip(start, end)]

        data = data[:, slices[0], slices[1]]

        return data

    def on_train_epoch_end(self, trainer, pl_module):
        preds = []
        for data, *_ in self.faults_dataloader:
            data = torch.unsqueeze(data, 1).to(pl_module.device)
            batch_preds = pl_module.network(data).argmax(dim=1)
            preds.append(batch_preds)
        preds = torch.cat(preds, dim=0).to("cpu")
        preds = self.extract_patch(preds)

        class_labels = {i: name for i, name in enumerate(pl_module.class_names)}
        for i, (sample, day, issue, center, split) in enumerate(
            zip(self.samples, self.days, self.issues, self.centers, self.splits)
        ):
            plot = wandb.Image(
                self.data[i].numpy(),
                caption=f"Sample: {sample}, Day: {day}, Issue: {issue} Center: {center} ({split})",
                masks={
                    "predictions": {"mask_data": preds[i].numpy(), "class_labels": class_labels},
                    "labels": {"mask_data": self.labels[i].numpy(), "class_labels": class_labels},
                },
            )

            wandb.log({f"faults/{issue}/{sample}/{center}": plot})
