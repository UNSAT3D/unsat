from data import create_dataloaders
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from models import UltraLocalModel
import torch
import torch.nn.functional as F

dataloaders = create_dataloaders(
    hdf5_path='../../data/data.h5',
    train_samples=['maize/coarse/loose', 'maize/fine/dense'],
    height_range=(400, 1_200),
    train_day_range=(0, 7),
    validation_split=0.1,
    batch_size=8,
)

model = UltraLocalModel(input_size=1, hidden_sizes=[16, 32, 32], output_size=5)


class LTest(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y = batch
        # labels shouldn't have the channels dimension
        y = torch.reshape(y, y.shape[:-1])
        y_hat = self.model(x)
        y_hat = torch.reshape(y_hat, (-1, 5))
        y = torch.reshape(y, (-1,))
        loss = F.cross_entropy(y_hat, y)
        self.log("train/loss", loss)
        print(f"loss: {loss}")
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)


model_L = LTest(model)
logger = WandbLogger(project="local-model")

trainer = L.Trainer(log_every_n_steps=1, accelerator='cpu', logger=logger)
trainer.fit(model_L, dataloaders['train'], dataloaders['val'])
