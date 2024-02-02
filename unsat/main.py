from data import XRayDataModule
from lightning.pytorch.cli import LightningCLI
from train import LightningTrainer, WandbSaveConfigCallback


def cli_main():
    """
    Standard lightning CLI definition, with added callback to save the lightning config to wandb.

    Usage e.g.: python main.py fit -c config.yaml
    See also https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_intermediate.html
    """
    cli = LightningCLI(
        model_class=LightningTrainer,
        datamodule_class=XRayDataModule,
        save_config_callback=WandbSaveConfigCallback,
        save_config_kwargs={"save_to_log_dir": False, "config_filename": "lightning_config.yaml"},
    )


if __name__ == "__main__":
    cli_main()
