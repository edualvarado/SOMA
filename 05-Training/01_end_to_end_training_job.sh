#!/bin/bash
#SBATCH -p gpu20                 # <--- CHANGE THIS to your cluster's GPU partition name (e.g., 'gpu', 'gpu20', 'rtx6000')
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH -c 16                    # 16 CPU Cores (Sufficient for num_workers=4)
#SBATCH -t 2-00:00:00            # Time limit: 2 days
#SBATCH --mem=32G                # Memory: 32GB
#SBATCH --signal=B:SIGTERM@120   # Handler for clean exit
#SBATCH -o slurm_train-%j.out    # Log output file

# ==============================================================================
# Script to run SOMA End-to-End Training on Slurm.
# ==============================================================================

# JOBS
# ====


# ====

# 1. Setup Environment
# --------------------
eval "$(conda shell.bash hook)"
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

# Activate your specific environment
conda activate musk

# Enable errexit to stop on errors
set -e

# 2. Optimization flags 
# ---------------------
# These prevent numpy/pytorch from spawning too many threads per process,
# which can cause contention when using a DataLoader.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# 3. Directory Setup
# ------------------
# Ensure we run in the directory where the .sh script (and likely the .py file) is located.
cd $SLURM_SUBMIT_DIR

echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "Dir : $(pwd)"
echo "=================================================="

# 4. Run Training
# ---------------
# Using 'python -u' allows stdout to be unbuffered (better for real-time log monitoring)
python -u 01_end_to_end_training.py

echo "=================================================="
echo "Training Finished."
echo "=================================================="