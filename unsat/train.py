import lightning as L
from lightning.pytorch.cli import OptimizerCallable, SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger
import torch
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, ConfusionMatrix, F1Score
from torchmetrics.wrappers import ClasswiseWrapper

import wandb


class LightningTrainer(L.LightningModule):
    def __init__(
        self,
        network,
        class_names,
        dimension,
        input_channels,
        optimizer: OptimizerCallable,
        **kwargs,
    ):
        """
        Lightning module defining the network and the training loop.

        Args:
            network (nn.Module):
                The network to train.
            class_names (List[str]):
                The names of the classes.
            dimension (int):
                The number of spatial dimensions.
            input_channels (int):
                The number of input channels.
        """
        torch.autograd.set_detect_anomaly(True)
        super().__init__()
        self.optimizer = optimizer

        self.class_names = class_names
        self.num_classes = len(class_names)

        self.network = network
        self.network.dimension = dimension
        self.network.num_classes = self.num_classes
        self.network.input_channels = input_channels
        self.network.build()

        metrics_args = dict(task="multiclass", num_classes=self.num_classes, ignore_index=-1)
        self.metrics = torch.nn.ModuleDict()
        self.metrics['acc'] = torch.nn.ModuleDict(
            {
                'train_': Accuracy(**metrics_args, average='macro'),
                'val_': Accuracy(**metrics_args, average='macro'),
            }
        )
        self.metrics['f1'] = torch.nn.ModuleDict(
            {
                'train_': F1Score(**metrics_args, average='macro'),
                'val_': F1Score(**metrics_args, average='macro'),
            }
        )
        self.metrics['acc_per_class'] = torch.nn.ModuleDict(
            {
                'train_': ClasswiseWrapper(
                    Accuracy(**metrics_args, average=None), labels=self.class_names
                ),
                'val_': ClasswiseWrapper(
                    Accuracy(**metrics_args, average=None), labels=self.class_names
                ),
            }
        )
        self.metrics['f1_per_class'] = torch.nn.ModuleDict(
            {
                'train_': ClasswiseWrapper(
                    F1Score(**metrics_args, average=None), labels=self.class_names
                ),
                'val_': ClasswiseWrapper(
                    F1Score(**metrics_args, average=None), labels=self.class_names
                ),
            }
        )

        metrics_args['normalize'] = 'true'
        self.metrics['confusion'] = torch.nn.ModuleDict(
            {'train_': ConfusionMatrix(**metrics_args), 'val_': ConfusionMatrix(**metrics_args)}
        )

        # These can be overriden to represent class frequencies by using the ClassWeightsCallback
        self.class_weights = torch.ones(self.num_classes)

    def training_step(self, batch, batch_idx):
        x, labels, mask = batch  # labels shape (batch_size, X, Y)
        preds = self.network(x)  # (batch_size, C, X, Y)

        loss = self.compute_loss(preds, labels, mask)
        self.log("train/loss", loss)

        self.compute_metrics(preds, labels, mask, mode="train_")

        return loss

    def validation_step(self, batch, batch_idx):
        x, labels, mask = batch
        preds = self.network(x)

        loss = self.compute_loss(preds, labels, mask)
        self.log("val/loss", loss)

        self.compute_metrics(preds, labels, mask, mode="val_")

    def compute_loss(self, preds, labels, mask):
        return _compute_loss(preds, labels, mask, self.class_weights)

    def compute_metrics(self, preds, labels, mask, mode):
        # Replace patch border, if set, with -1 in labels.
        masked_labels = torch.where(mask, labels, -1)

        acc_overall = self.metrics['acc'][mode](preds, masked_labels)
        self.log(f"{mode[:-1]}/acc/all", self.metrics['acc'][mode], on_step=True, on_epoch=True)

        self.metrics['f1'][mode](preds, masked_labels)
        self.log(f"{mode[:-1]}/f1", self.metrics['f1'][mode], on_step=True, on_epoch=True)

        accs_per_class = self.metrics['acc_per_class'][mode](preds, masked_labels)
        accs_per_class = {
            f"{mode[:-1]}/acc/{k.split('_')[1]}": v for k, v in accs_per_class.items()
        }
        wandb.log(accs_per_class)

        f1_per_class = self.metrics['f1_per_class'][mode](preds, masked_labels)
        f1_per_class = {f"{mode[:-1]}/f1/{k.split('_')[1]}": v for k, v in f1_per_class.items()}
        wandb.log(f1_per_class)

        if (self.current_epoch % 100 == 0) and (self.current_epoch > 0):
            self.compute_confusion(preds, masked_labels, mode)

    def compute_confusion(self, preds, labels, mode):
        confusion = self.metrics['confusion'][mode](preds, labels)

        data = []
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                data.append([self.class_names[i], self.class_names[j], confusion[i, j].item()])

        columns = ["Actual", "Predicted", "nPredictions"]
        fields = {k: k for k in columns}
        plot = wandb.plot_table(
            "wandb/confusion_matrix/v1",
            wandb.Table(columns=columns, data=data),
            fields,
            {"title": f"{mode[:-1]} epoch {self.current_epoch}"},
        )
        wandb.log({f"confusion/{mode[:-1]}": plot})

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        return optimizer


def _compute_loss(preds, labels, mask, class_weights):
    per_pixel_losses = F.cross_entropy(preds, labels, weight=class_weights, reduction='none')
    per_pixel_losses = per_pixel_losses * mask
    loss = per_pixel_losses.sum() / mask.sum()
    return loss


class WandbSaveConfigCallback(SaveConfigCallback):
    """
    Custom callback to save the lightning config to wandb.
    """

    def save_config(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str):
        logger = trainer.logger
        save_dir = logger.experiment.dir
        print("save dir:", save_dir)
        config_path = f"{save_dir}/{self.config_filename}"
        self.parser.save(
            self.config,
            config_path,
            skip_none=False,
            overwrite=self.overwrite,
            multifile=self.multifile,
        )
        logger.experiment.config['lightning_config'] = self.config

        # save model config separately
        model_name = self.config.model.network.class_path.split('.')[-1]
        logger.experiment.config['model'] = model_name
        for name, val in self.config.model.network.init_args.items():
            logger.experiment.config[name] = val
        optimizer_name = self.config.model.optimizer.class_path.split('.')[-1]
        logger.experiment.config['optimizer'] = optimizer_name
        logger.experiment.config['lr'] = self.config.model.optimizer.init_args.lr
        for name, val in self.config.model.optimizer.init_args.items():
            logger.experiment.config[f"opt/{name}"] = val
