![Python package](https://github.com/UNSAT3D/unsatIO/workflows/Install%20and%20test%20Python%20package/badge.svg)
[![codecov](https://codecov.io/gh/UNSAT3D/unsatIO/graph/badge.svg)](https://codecov.io/gh/UNSAT3D/unsatIO)

# Input/Output tools for UNSAT

## Installation

### Using `poetry` (recommended)

1. Navigate to the project folder
2. Run `poetry install`

Your working environment can be activated via `poetry shell`.
More information on poetry available [here](https://python-poetry.org/).

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