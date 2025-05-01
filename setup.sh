#!/bin/bash

# This script will set up the Conda environment and install the dependencies

export PROJECT_NAME="zwgrad"
CONDA_ENV_NAME="${PROJECT_NAME}_env"
CONDA_ENV_FILE="environment.yml"
PYTHON_VERSION="3.11"

# Check if the conda environment exists
if conda env list | grep -q "$CONDA_ENV_NAME"; then
    echo "Conda environment '$CONDA_ENV_NAME' already exists. Updating it..."
    conda env update --name "$CONDA_ENV_NAME" --file $CONDA_ENV_FILE --prune
else
    echo "Conda environment '$CONDA_ENV_NAME' not found. Creating it from environment.yml..."
    conda env create --name "$CONDA_ENV_NAME" --file $CONDA_ENV_FILE
fi

# Activate the environment
conda activate "$CONDA_ENV_NAME"

echo "Environment setup complete."

# General environment variables
export PROJECT_ROOT=$PWD
export SRC_ROOT="${PROJECT_ROOT}/${PROJECT_NAME}/"

export PYTHONPATH=$SRC_ROOT