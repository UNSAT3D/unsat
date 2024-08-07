#!/bin/bash
# Script to run a fit job on the cluster
# Usage: `sbatch run.slurm <lightning CLI options>`
#
# NOTE:
# These settings are for minimal tests.
# Make sure to adjust the `--time` parameter to the expected runtime of your job
# Also be aware that the gpu_mig partition uses only half a GPU. If you get out of memory errors,
# try the gpu partition instead.
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --partition=gpu_mig
#SBATCH --gpus-per-node=1
#SBATCH --time=10:10:00
#SBATCH --output=logs_slurm/%j.out

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source /home/sciarella1/.virtualenvs/unsat/bin/activate

poetry run python unsat/main.py fit -c "$@"
