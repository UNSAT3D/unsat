import lightning as L
from lightning.pytorch.cli import OptimizerCallable, SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger
from models import UltraLocalModel
import torch
import torch.nn.functional as F
from torchmetrics.classification import Accuracy

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
        try:
            self.model = MODEL_CLASSES[model_class](
                **model_kwargs, input_size=1, output_size=self.num_classes
            )
        except KeyError:
            raise ValueError(f"Model class {model_class} not found.")

        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)

    def training_step(self, batch, batch_idx):
        x, labels = batch
        preds = self.model(x)

        loss = self.compute_loss(preds, labels)
        self.log("train/loss", loss)

        self.train_acc(preds.reshape(-1, self.num_classes), labels.reshape(-1))
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        preds = self.model(x)

        loss = self.compute_loss(preds, labels)

        self.log("val/loss", loss)

        self.val_acc(preds.reshape(-1, self.num_classes), labels.reshape(-1))
        self.log("val/acc", self.val_acc, on_step=True, on_epoch=True)

    def compute_loss(self, preds, labels):
        preds = torch.reshape(preds, (-1, self.num_classes))
        labels = torch.reshape(labels, (-1,))
        loss = F.cross_entropy(preds, labels)
        return loss

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
