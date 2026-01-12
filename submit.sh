#!/bin/bash
#SBATCH --job-name=entail_optuna
#SBATCH --output=entail_optuna_%j.out
#SBATCH --error=entail_optuna_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48G

# 1. Load Anaconda
module load python/anaconda-2023.03

# 2. Setup a local package folder to avoid conflicts
export PYTHONUSERBASE=$PWD/.local_tmp
mkdir -p $PYTHONUSERBASE
export PATH=$PYTHONUSERBASE/bin:$PATH

# 3. Install Dependencies
# We do NOT use the CPU index-url here, so Torch will use the GPU.
echo "Installing dependencies..."
pip install --user --no-warn-script-location \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    transformers==4.34.0 \
    sentence-transformers==2.7.0 \
    pandas \
    openpyxl \
    scikit-learn \
    optuna==3.5.0 \
    tqdm \
    plotly \
    papermill \
    ipykernel

# 4. Run the Notebook
# We force the kernel to 'python3' to use the environment we just built.
echo "Starting Papermill..."
papermill FreeEntailmentAlgorithm.ipynb results_graphs_optuna.ipynb -k python3