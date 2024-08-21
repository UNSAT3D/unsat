[![Python package](https://github.com/UNSAT3D/unsat/workflows/Install%20and%20test%20Python%20package/badge.svg)](https://github.com/UNSAT3D/unsat/actions/workflows/python.yaml)
[![codecov](https://codecov.io/gh/UNSAT3D/unsat/graph/badge.svg)](https://codecov.io/gh/UNSAT3D/unsat)

![header](https://capsule-render.vercel.app/api?type=egg&height=200&color=0:D98F61,100:BD6629&text=UNSAT%20🌱&textBg=false&section=header&reversal=false&animation=scaleIn&strokeWidth=3&stroke=95d6a4&desc=AI%20analysis%20tool%20for%20rooted%20soil&fontColor=3D824a&descSize=35&fontAlign=50&fontAlignY=31&descAlignY=50)

Sure! Here’s an enhanced and structured version of your README content, tailored for documentation purposes. The content is split into different sections/pages, with a more verbose and structured introduction.

---

# UNSAT 🌱 Documentation

Welcome to the **UNSAT** documentation! This guide provides a comprehensive overview of the UNSAT project, an AI analysis tool designed for rooted soil analysis. It will walk you through installation, usage, and other key aspects, ensuring you can leverage UNSAT effectively.

## Overview

**UNSAT** is a Python-based tool tailored for analyzing and modeling rooted soils using AI techniques. It utilizes advanced configurations and integrates with tools like Weights and Biases (wandb) for experiment tracking and model evaluation. This documentation is divided into several sections to help you navigate and use UNSAT with ease.

---

## Table of Contents

1. **[Introduction to UNSAT](#introduction-to-unsat)**
2. **[Installation Guide](#installation-guide)**
   - [Using Poetry (Recommended)](#using-poetry-recommended)
   - [Snellius Installation](#snellius-installation)
3. **[Usage Instructions](#usage-instructions)**
   - [Running Experiments](#running-experiments)
   - [Weights and Biases Setup](#weights-and-biases-setup)
4. **[Configuration Management](#configuration-management)**
   - [Overriding Configurations](#overriding-configurations)
   - [Profiling Options](#profiling-options)
5. **[Contributing to UNSAT](#contributing-to-unsat)**
   - [Using the Linter](#using-the-linter)
   - [Contributing Guidelines](#contributing-guidelines)

---

## Introduction to UNSAT

**UNSAT** is a specialized tool for analyzing soil structures, particularly focusing on rooted soils using machine learning and AI methodologies. The tool is designed to be flexible, allowing for a variety of configurations and setups to match different experimental needs. It integrates seamlessly with Weights and Biases for tracking experiments, and it is optimized to run efficiently on both local and remote environments, including HPC systems like Snellius.

---

{!../docs/installation.md!}

---

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

---

## Configuration Management

UNSAT is highly configurable, allowing you to tailor the system's behavior to your specific needs. All configurations are managed via YAML files, which can be easily edited or overridden. To understand and tailor your model configuration refer to [this page](https://unsat3d.github.io/unsat/config/) of the manual of *unsat*.


### Overriding Configurations

You can override default configurations by passing additional config files or command-line arguments. For example, to specify a different profiler, you can run:

```bash
poetry run python unsat/main.py fit -c configs/test_config.yaml --trainer.profiler pytorch
```

### Profiling Options

Profiling your runs can help you optimize performance. UNSAT supports multiple profiling tools and configurations. To enable a predefined profiler, use the following command:

```bash
poetry run python unsat/main.py fit -c configs/profiler.yaml
```

You can mix and match configurations as needed to achieve the desired results.

---

## Snellius Usage

To run experiments on Snellius, use the provided SLURM script. Here's how:

1. **Submit a Job**:
   From the top-level project directory, execute:
   ```bash
   sbatch scripts/run.slurm configs/test_config.yaml
   ```

2. **Check Outputs**:
   After the job completes, various outputs will be generated:
   - **logs_slurm/**: Contains terminal outputs.
   - **wandb/**: Metadata synchronized with Weights and Biases.
   - **project-unsat/**: Model checkpoints and other locally stored data.

For more advanced usage and troubleshooting on Snellius, consult the [Snellius Documentation](https://www.surf.nl/en/knowledgebase).

---

## Contributing to UNSAT

We welcome contributions from the community! To maintain code quality and consistency, please adhere to the following guidelines.

### Using the Linter

Our project uses a linter to enforce code quality. The linter runs automatically on each commit to GitHub. If you'd like to run the linter locally:

1. **Activate Your Environment**:
   ```bash
   poetry shell
   ```

2. **Install Pre-commit**:
   ```bash
   pre-commit install
   ```

The linter will automatically run after each commit, and it will suggest or apply fixes where necessary. Note that you may need to stage and commit the changes again after the linter has made adjustments.

### Contributing Guidelines

Please follow our [Contribution Guidelines](link-to-contributing-guide) when submitting pull requests. Ensure your code is well-documented, tested, and adheres to the project's coding standards.