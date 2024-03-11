import lightning as L
from lightning.pytorch.cli import OptimizerCallable, SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger
from models import UltraLocalModel
import torch
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, ConfusionMatrix, F1Score
from torchmetrics.wrappers import ClasswiseWrapper

import wandb

MODEL_CLASSES = {"ultra_local": UltraLocalModel}


class LightningTrainer(L.LightningModule):
    def __init__(
        self, model_class: str, model_kwargs: dict, optimizer: OptimizerCallable, **kwargs
    ):
        """
        Lightning module defining the model and the training loop.

        Args:
            model_class (str):
                The model class to use.
            model_kwargs (dict):
                The keyword arguments to pass to the model class.
        """
        super().__init__()
        self.optimizer = optimizer

        self.num_classes = 5
        self.class_names = ["water", "-", "air", "root", "soil"]

        try:
            self.model = MODEL_CLASSES[model_class](**model_kwargs, num_classes=self.num_classes)
        except KeyError:
            raise ValueError(f"Model class {model_class} not found.")

        metrics_args = dict(task="multiclass", num_classes=self.num_classes)
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

        metrics_args = dict(task="multiclass", num_classes=self.num_classes, normalize='all')
        self.metrics['confusion'] = torch.nn.ModuleDict(
            {'train_': ConfusionMatrix(**metrics_args), 'val_': ConfusionMatrix(**metrics_args)}
        )

        freqs = torch.Tensor([0.3140, 0.2611, 0.0402, 0.0013, 0.3833])
        weight = 1 / freqs
        weight = weight / weight.sum()
        self.loss_class = torch.nn.CrossEntropyLoss(weight=weight)

    def training_step(self, batch, batch_idx):
        x, labels = batch  # labels shape (batch_size, X, Y)
        preds = self.model(x)  # (batch_size, C, X, Y)

        loss = self.compute_loss(preds, labels)
        self.log("train/loss", loss)

        self.compute_metrics(preds, labels, mode="train_")

        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        preds = self.model(x)

        loss = self.compute_loss(preds, labels)
        self.log("val/loss", loss)

        self.compute_metrics(preds, labels, mode="val_")

    def compute_loss(self, preds, labels):
        return self.loss_class(preds, labels)

    def compute_metrics(self, preds, labels, mode):
        acc_overall = self.metrics['acc'][mode](preds, labels)
        self.log(f"{mode[:-1]}/acc/all", self.metrics['acc'][mode], on_step=True, on_epoch=True)

        self.metrics['f1'][mode](preds, labels)
        self.log(f"{mode[:-1]}/f1", self.metrics['f1'][mode], on_step=True, on_epoch=True)

        accs_per_class = self.metrics['acc_per_class'][mode](preds, labels)
        accs_per_class = {
            f"{mode[:-1]}/acc/{k.split('_')[1]}": v for k, v in accs_per_class.items()
        }
        wandb.log(accs_per_class)

        f1_per_class = self.metrics['f1_per_class'][mode](preds, labels)
        f1_per_class = {f"{mode[:-1]}/f1/{k.split('_')[1]}": v for k, v in f1_per_class.items()}
        wandb.log(f1_per_class)

        if self.current_epoch % 100 == 0:
            self.compute_confusion(preds, labels, mode)

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
