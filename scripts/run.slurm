#!/bin/bash
# Script to run a fit job on the cluster
# Usage: `sbatch run.slurm <lightning CLI options>`
#
# NOTE: Make sure to adjust the `--time` parameter to the expected runtime of your job
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=logs_slurm/%j.out

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

poetry run python unsat/main.py fit -c "$@"
