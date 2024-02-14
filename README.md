[![Python package](https://github.com/UNSAT3D/unsat/workflows/Install%20and%20test%20Python%20package/badge.svg)](https://github.com/UNSAT3D/unsat/actions/workflows/python.yaml)
[![codecov](https://codecov.io/gh/UNSAT3D/unsat/graph/badge.svg)](https://codecov.io/gh/UNSAT3D/unsat)

# UNSAT

## Installation

### Using `poetry` (recommended)

0. Running on Snellius? Read first [the subsection](#snellius) below
1. Clone repo: `git clone git@github.com:UNSAT3D/unsat.git`
2. Navigate to the project folder: `cd unsat`
3. If necessary install poetry: `pip install poetry`
4. Run `poetry install`

Your working environment can be activated via `poetry shell`, or to use it in a single command, prefix it with `poetry run`.
More information on poetry available [here](https://python-poetry.org/).

To access weights and biases, run `poetry run wandb login` once, and copy your API key found [here](https://wandb.ai/authorize).

### Weights and biases

We use weights and biases to track our experiments. 
Make sure this is set up first.

## Usage

Once installed, from the project folder run

```bash
poetry run python unsat/main.py fit -c configs/test_config.yaml --data.hdf5_path <path to data>
```

This does a short training run and uploads the results to weights and biases.
In the terminal you should see a link to the run.
The config used will be saved to weights and biases too as `lightning_config.yaml`.

To do more useful runs, look at other config files or modify it yourself.
All configuration settings should be in the config file rather than the code itself.

Anything in one config can be overridden by a second one, or by single options as we do above for the data path.
For instance to turn on profiling you can add: `--trainer.profiler pytorch`, or to use a predefined
configured profiler, add `configs/profiler.yaml`.

## Snellius

On snellius, to install, run these commands:
```bash
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
```
and then follow the general instructions at the top.

To submit a job, a basic script is provided, simply run (from the top project folder):
```bash
sbatch scripts/run.slurm configs/test_config.yaml
```

This will create several outputs:
- In `logs_slurm/`: The terminal output
- In `wandb/`: The metadata of the run which is synced to weights and biases.
- In `project-unsat/`: Model checkpoints and other data only stored locally.

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
