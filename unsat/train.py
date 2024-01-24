import lightning as L
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger
from models import UltraLocalModel
import torch
import torch.nn.functional as F
from torchmetrics.classification import Accuracy

MODEL_CLASSES = {"ultra_local": UltraLocalModel}


class LightningTrainer(L.LightningModule):
    def __init__(self, model_class: str, model_kwargs: dict, **kwargs):
        """
        Hello world!
        """
        super().__init__()

        self.num_classes = 5
        self.model = MODEL_CLASSES[model_class](
            **model_kwargs, input_size=1, output_size=self.num_classes
        )

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
        return torch.optim.Adam(self.model.parameters(), lr=1e-1)
