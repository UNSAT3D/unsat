## Usage Instructions

Once you've installed UNSAT, you're ready to start running experiments. This section will guide you through the process.

### Running Experiments

To run a basic experiment with UNSAT, navigate to the project directory and execute the following command:

```bash
poetry run python unsat/main.py fit -c configs/test_config.yaml --data.hdf5_path <path to data>
```

This command will start a short training session using the specified configuration file and data path. The results will be automatically uploaded to Weights and Biases. To understand and tailor your model configuration refer to [this page](https://unsat3d.github.io/unsat/config/) of the manual of *unsat*.

### Weights and Biases Setup

Weights and Biases (wandb) is a powerful tool for tracking machine learning experiments. To set up wandb with UNSAT:

1. **Login to wandb**:
   ```bash
   poetry run wandb login
   ```

2. **Enter Your API Key**:
   You can find your API key [here](https://wandb.ai/authorize). Paste it when prompted.

For more detailed guidance, visit the [Weights and Biases Quickstart Guide](https://docs.wandb.ai/quickstart).

