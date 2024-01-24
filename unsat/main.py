from data import XRayDataModule
from lightning.pytorch.cli import LightningCLI
from train import LightningTrainer


def cli_main():
    cli = LightningCLI(
        model_class=LightningTrainer, datamodule_class=XRayDataModule, save_config_callback=None
    )


if __name__ == "__main__":
    cli_main()
