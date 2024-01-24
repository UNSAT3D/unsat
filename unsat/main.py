from data import XRayDataModule
from lightning.pytorch.cli import LightningCLI
from train import LightningTrainer, WandbSaveConfigCallback


def cli_main():
    cli = LightningCLI(
        model_class=LightningTrainer,
        datamodule_class=XRayDataModule,
        save_config_callback=WandbSaveConfigCallback,
        save_config_kwargs={"save_to_log_dir": False, "config_filename": "lightning_config.yaml"},
    )


if __name__ == "__main__":
    cli_main()
