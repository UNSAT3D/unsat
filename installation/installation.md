## Installation Guide

### Using Poetry (Recommended)

Poetry is a dependency management tool that helps you manage your Python environments and dependencies efficiently. We highly recommend using Poetry to set up UNSAT. Follow these steps to get started:

0. Running on Snellius? Read first [the subsection](#snellius) below

1. **Clone the Repository**:  
   First, clone the UNSAT repository to your local machine:
   ```bash
   git clone https://github.com/UNSAT3D/unsat.git
   ```
   
2. **Navigate to the Project Folder**:  
   Move into the cloned repository directory:
   ```bash
   cd unsat
   ```

3. **Install Poetry**:  
   If Poetry is not already installed on your system, you can install it using pip:
   ```bash
   pip install poetry
   ```

4. **Install Dependencies**:  
   Run the following command to install all required dependencies:
   ```bash
   poetry install
   ```

5. **Activate the Environment**:  
   To work within the UNSAT environment, activate the virtual environment created by Poetry:
   ```bash
   poetry shell
   ```
   Alternatively, you can run commands within this environment by prefixing them with `poetry run`.

For more details on Poetry, visit the [Poetry Documentation](https://python-poetry.org/).

### Snellius Installation

If you're working on the Snellius supercomputer, you'll need to load specific modules before installation. Follow these steps:

1. **Load Required Modules**:
   ```bash
   module load 2023
   module load Python/3.11.3-GCCcore-12.3.0
   ```

2. **Install UNSAT**:
   After loading the modules, follow the general installation steps as described in the [Poetry Installation Guide](#using-poetry-recommended).

For detailed instructions on submitting jobs on Snellius, refer to the [Snellius Usage Section](#snellius-usage).

