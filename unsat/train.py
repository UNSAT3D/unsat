from data import create_dataloaders
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from models import UltraLocalModel
import torch
import torch.nn.functional as F
from torchmetrics.classification import Accuracy

dataloaders = create_dataloaders(
    hdf5_path='../../data/data.h5',
    train_samples=['maize/coarse/loose', 'maize/fine/dense'],
    height_range=(800, 1_200),
    train_day_range=(0, 3),
    validation_split=0.1,
    batch_size=8,
)

model = UltraLocalModel(input_size=1, hidden_sizes=[16, 16], output_size=5)


class LTest(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

        self.num_classes = 5

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


model_L = LTest(model)
logger = WandbLogger(project="local-model")

trainer = L.Trainer(
    log_every_n_steps=1, accelerator='cpu', logger=logger, val_check_interval=3, max_epochs=4
)
trainer.fit(model_L, dataloaders['train'], dataloaders['val'])
