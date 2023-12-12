import lightning as L
from lightning.pytorch.loggers import WandbLogger
from models import UltraLocalModel
import torch
import torch.nn.functional as F

from data import create_dataloaders

dataloaders = create_dataloaders(
    hdf5_path='../data/data.h5',
    train_samples=['maize/coarse/loose', 'maize/fine/dense'],
    height_range=(400, 1_200),
    train_day_range=(0, 7),
    validation_split=0.1,
)

model = UltraLocalModel(input_size=1, hidden_sizes=[16, 32, 32], output_size=5)


class LTest(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")

    def training_step(self, batch, batch_idx):
        x, y = batch
        # convert y to one-hot
        y = torch.nn.functional.one_hot(y, num_classes=5).type(torch.float32)
        y_hat = self.model(x)
        y = torch.reshape(y, (-1, 5))
        y_hat = torch.reshape(y_hat, (-1, 5))
        loss = self.loss(y, y_hat)
        self.log("train/loss", loss)
        print(f"loss: {loss}")
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)


model_L = LTest(model)
logger = WandbLogger(project="UNSAT", name="myrun")

trainer = L.Trainer(log_every_n_steps=10, accelerator='cpu', logger=logger)
trainer.fit(model_L, dataloaders['train'])
