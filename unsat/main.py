from data import XRayDataModule
from lightning.pytorch.cli import LightningCLI
from train import LightningTrainer, WandbSaveConfigCallback
from callbacks import ClassWeightsCallback


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.class_names", "model.class_names")
        parser.link_arguments("data.dimension", "model.dimension")
        parser.link_arguments("data.input_channels", "model.input_channels")


def cli_main():
    """
    Standard lightning CLI definition, with added callback to save the lightning config to wandb.

    Usage e.g.: python main.py fit -c config.yaml
    See also https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_intermediate.html
    """
    cli = MyLightningCLI(
        model_class=LightningTrainer,
        datamodule_class=XRayDataModule,
        class_weights_callback=ClassWeightsCallback,
        save_config_callback=WandbSaveConfigCallback,
        save_config_kwargs={"save_to_log_dir": False, "config_filename": "lightning_config.yaml"},
        auto_configure_optimizers=False,
    )


if __name__ == "__main__":
    cli_main()
