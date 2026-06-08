#!/bin/bash
#SBATCH -p cpu20
#SBATCH -c 16
#SBATCH -t 6-23:00:00
#SBATCH --signal=B:SIGTERM@120
#SBATCH -o slurm_eval-%j.out

# ==============================================================================
# Script to run HIT Bio Evaluation on Slurm.
# Results (JSON/PNG) will be saved in the directory where you run sbatch.
# ==============================================================================

# ==============================================================================
# SUMMARY OF JOBS 
# 47060088: S1, shot_001 -> R
# 47060097: S2, shot_001 -> R
# 47060113: S3, shot_001 -> R
# 47060126: S4, shot_001 -> R
# 47060193: S5, shot_001 -> R
# ==============================================================================


# 1. Setup Environment
eval "$(conda shell.bash hook)"
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

# Activate your specific environment
conda activate musk

# Enable errexit to stop on errors
set -e

# 2. Optimization flags (Single thread per task to avoid numpy explosion)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# 3. Ensure we run in the directory where the .sh script is located
# This guarantees the .json and .png are saved "here"
cd $SLURM_SUBMIT_DIR

echo "=================================================="
echo "Job Started on Node: $SLURMD_NODENAME"
echo "Working Directory: $(pwd)"
echo "Running Evaluation..."
echo "=================================================="

# 4. Run the Python Script
# Assumes hit_bio_evaluation.py is in the same folder
python hit_bio_evaluation.py

echo "=================================================="
echo "Evaluation Complete."
echo "Check $(pwd) for .json and .png outputs."
echo "=================================================="