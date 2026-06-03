#!/bin/bash --login

#SBATCH -p gpuA              # A100 (80GB) GPUs  [up to 12 CPU cores per GPU permitted]
### Required flags
#SBATCH -G 1                 # (or --gpus=N) Number of GPUs
#SBATCH -t 1-0               # Wallclock timelimit (1-0 is one day, 4-0 is max permitted)

# Latest version of CUDA
if command -v module >/dev/null 2>&1; then
    module load libs/cuda
fi

echo "Job is using ${NGPUS:-1} GPU(s) with ID(s) ${CUDA_VISIBLE_DEVICES:-unset} and ${NSLOTS:-unset} CPU core(s)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -n "${VENV_PATH:-}" ] && [ -f "${VENV_PATH}/bin/activate" ]; then
    # Optional venv activation for cluster jobs.
    source "${VENV_PATH}/bin/activate"
elif [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi

# Default values arg parameters
DEFAULT_CONFIG="${PROJECT_ROOT}/configs/ucf-cap.json"
DEFAULT_SAVE_DIR="${PROJECT_ROOT}/data"
DEFAULT_NAME="Eval"
DEFAULT_CHECKPOINT="${PROJECT_ROOT}/data/models/best_multimodal_model.pth"

# Usage function to display help
usage() {
    echo "Usage: $0 [--config <config_path>] [--checkpoint <checkpoint_path>] [--save_dir <save_dir>] [--name <name>]"
    exit 1
}

# Initialize variables with default values
CONFIG="$DEFAULT_CONFIG"
SAVE_DIR="$DEFAULT_SAVE_DIR"
NAME="$DEFAULT_NAME"
CHECKPOINT="$DEFAULT_CHECKPOINT"


# Parse command line arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --save_dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --name)
            NAME="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

# Run the Python script with the provided or default arguments
python3 "${PROJECT_ROOT}/src/multi_vit/eval_multimodal.py" --checkpoint="$CHECKPOINT" --save_dir="$SAVE_DIR" --config="$CONFIG" --name="$NAME"
