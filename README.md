[![Python package](https://github.com/UNSAT3D/unsat/workflows/Install%20and%20test%20Python%20package/badge.svg)](https://github.com/UNSAT3D/unsat/actions/workflows/python.yaml)
[![codecov](https://codecov.io/gh/UNSAT3D/unsat/graph/badge.svg)](https://codecov.io/gh/UNSAT3D/unsat)

# UNSAT

## Installation

### Using `poetry` (recommended)

1. Navigate to the project folder
2. Run `poetry install`

Your working environment can be activated via `poetry shell`.
More information on poetry available [here](https://python-poetry.org/).

### Weights and biases

We use weights and biases to track our experiments. 
Make sure this is set up first.

## Usage

Once installed, from the project folder run

```bash
python unsat/main.py fit -c configs/test_config.yaml --data.hdf5_path <path to data>
```

This does a short training run and uploads the results to weights and biases.
In the terminal you should see a link to the run.
The config used will be saved to weights and biases too as `lightning_config.yaml`.

To do more useful runs, look at other config files or modify it yourself.
All configuration settings should be in the config file rather than the code itself.

Anything in one config can be overridden by a second one, or by single options as we do above for the data path.
For instance to turn on profiling you can add: `--trainer.profiler pytorch`.

## Contributing

### Linter

#### Remote

The linter runs automatically every time you push a commit to GitHub.

#### Local
If you want to use the linter locally, you'll have to install it manually:

1. Activate your working environment (typically via `poetry shell`).
2. Install `pre-commit` (by running `pre-commit install`).

The linter will be executed after each commit.
The linting will be performed automatically in case it is needed.
The affected files will **need staging and commiting again**.
