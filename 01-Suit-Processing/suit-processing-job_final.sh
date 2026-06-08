#!/bin/bash
#SBATCH -p cpu20
#SBATCH -c 64
#SBATCH -t 6-23:00:00
#SBATCH --signal=B:SIGTERM@120
#SBATCH -o /scratch/inf0/user/ealvarad/slurm-%j.out

# ==============================================================================
# Master script to process multiple shots in a single Slurm job.
#
# Usage: 

# squeue -u ealvarad

#-- VALENTIN (ALL DONE)

# sbatch suit-processing-job_final.sh --folder /CT/SOMA/static00/data/Valentin-12-12-25/ --board /CT/SOMA/work/01-Suit-Processing/configs/suits/charuco-suit.json

# tail slurm-.out
# squeue -j 

#-- TIMO

# sbatch suit-processing-job_final.sh --folder /CT/SOMA/static00/data/Timothee-17-12-25/ --board /CT/SOMA/work/01-Suit-Processing/configs/suits/charuco-suit.json

# tail slurm-48096400.out
# squeue -j 48096400

#-- MONA

# sbatch suit-processing-job_final.sh --folder /CT/SOMA/static00/data/Mona-22-12-25/ --board /CT/SOMA/work/01-Suit-Processing/configs/suits/charuco-suit.json

# tail slurm-48096403.out
# squeue -j 48096403

#-- SARAH (ALL DONE)

# sbatch suit-processing-job_final.sh --folder /CT/SOMA/static00/data/Sarah-19-12-25/ --board /CT/SOMA/work/01-Suit-Processing/configs/suits/charuco-suit.json

# tail slurm-.out
# squeue -j 

# NOTE: --folder should now point to the PARENT directory containing shot_001, shot_002, etc.
# ==============================================================================


# Make conda available:
eval "$(conda shell.bash hook)"

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

# Activate a conda environment:
conda activate soma

# Enable the 'errexit' option
set -e

# --- CRITICAL OPTIMIZATION: THREAD CONTROL ---
# Force mathematical libraries to be single-threaded
# This prevents each process from trying to spawn 128 threads, which kills performance.
# Since we launch 32 separate processes, we want each one to be lightweight.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "=================================================="
echo "Optimization Config:"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Cores per node: $SLURM_CPUS_ON_NODE"
echo "Total Cores Allocated: $SLURM_CPUS_PER_TASK"
echo "=================================================="

# --- Configuration ---

# Define the list of shots to process here

SHOTS_VALENTIN=()

SHOTS_TIMO=("002" "003" "004" "005" "006" "007" "008" "009" "010" "011" "012" "013")

SHOTS_MONA=("005" "006" "007" "008" "009" "010" "011" "012" "013" "014")

SHOTS_SARAH=()

# Set this to the subject being processed
SHOTS_TO_PROCESS=("${SHOTS_MONA[@]}")

# Initialize variables
base_folder="" 
board=""
output_folder=""

# Function to show usage
usage() {
  echo "Usage: $0 --folder <base_path_to_shots> --board <path>"
  exit 1
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --folder) base_folder="$2"; shift 2 ;;
    --board) board="$2"; shift 2 ;;
    --output-folder) output_folder="$2"; shift 2 ;; # Optional, usually specific to shot
    *) echo "Unknown parameter passed: $1"; usage ;;
  esac
done

# Check if base_folder is set
if [ -z "$base_folder" ]; then
  echo "Error: --folder argument is required (path to parent directory of shots)."
  exit 1
fi

# Helper Variable for board name
bname=$(basename $board .json)

# Hardcoded script folder as per your original file
cd /CT/SOMA/work/
script_folder="/CT/SOMA/work/01-Suit-Processing"

MAX_FRAMES=999999999
INTERPOLATION_2D_FRAMES=10
INTERPOLATION_3D_FRAMES=10
ISOLATION_3D_FRAMES=10

echo "=================================================="
echo "Master Job Started"
echo "Base Folder: $base_folder"
echo "Board: $board"
echo "Shots to process: ${SHOTS_TO_PROCESS[*]}"
echo "=================================================="

# --- Processing Loop ---
for shot in "${SHOTS_TO_PROCESS[@]}"; do
  
  # Construct the full path to the specific shot folder
  # ${base_folder%/} removes a trailing slash if present, ensuring path consistency
  target_folder="${base_folder%/}/shot_${shot}/"
  
  # Check if folder actually exists before trying to run python on it
  if [ ! -d "$target_folder" ]; then
      echo "WARNING: Folder $target_folder does not exist. Skipping..."
      continue
  fi

  echo ">>> Starting processing for: shot_${shot}"
  echo "    Target Folder: $target_folder"

  # --------------------------------
  # Detection
  # --------------------------------

  # 1. 2D Detections
  # We pass $target_folder to the python script
  # python $script_folder/detect_2D_final.py \
  #     --setup 3 \
  #     --modulus 1 \
  #     --board "$board" \
  #     --folder "$target_folder" \
  #     --parallel \
  #     --max_frames $MAX_FRAMES
  #     # --debug \

  # 2. 2D Interpolation
  # python $script_folder/interpolation_2D_final.py \
  #    --frames_2D_int $INTERPOLATION_2D_FRAMES \
  #    --folder "$target_folder" \

  # --------------------------------
  # Tracking
  # --------------------------------

  python $script_folder/tracking_3D_v2_final.py \
     --folder "$target_folder" \
     --frames_2D_int $INTERPOLATION_2D_FRAMES \
     --max_frames $MAX_FRAMES \

  python $script_folder/interpolation_3D_final.py \
     --frames_3D_int $INTERPOLATION_3D_FRAMES \
     --folder "$target_folder" \

  python $script_folder/post_process_triangulation_v3_final.py \
     --folder "$target_folder" \
     --frames_3D_int $INTERPOLATION_3D_FRAMES \
     --window_size $ISOLATION_3D_FRAMES \

  echo ">>> Finished processing: shot_${shot}"
  echo "--------------------------------------------------"

done

echo "All processing tasks in this job are complete."