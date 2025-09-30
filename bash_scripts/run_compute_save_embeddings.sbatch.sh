#!/bin/bash --login
#SBATCH -p gpuV               # v100 GPUs
#SBATCH -G 1                  # 1 GPU
#SBATCH -t 1-0               # Wallclock timelimit (1-0 is one day, 4-0 is max permitted)

# Latest version of CUDA
module load libs/cuda
echo "Job is using $NGPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $NSLOTS CPU core(s)"
source /mnt/iusers01/mace01/t08341gt/env_phd/bin/activate

# Default values arg parameters
DEFAULT_CONFIG="/mnt/iusers01/mace01/t08341gt/UCF_cap_mh/configs/ucf-cap.json"
DEFAULT_SAVE_DIR="/mnt/iusers01/mace01/t08341gt/UCF_cap_mh/data"
DEFAULT_MODEL_NAME="space_time_ciccio_13-03-25"
DEFAULT_NAME="TESTFRAMCESCP"

# Usage function to display help
usage() {
    echo "Usage: $0 [--config <config_path>] [--log_save <save_logs_path> ][--train <train_path>] [--save <save_path>] [--name <model_name>]"
    exit 1
}


# Initialize variables with default values
CONFIG="$DEFAULT_CONFIG"
SAVE_DIR="$DEFAULT_SAVE_DIR"
NAME="$DEFAULT_NAME"
MODEL_NAME="$DEFAULT_MODEL_NAME"
# Parse command line arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
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
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

# Run the Python script with the provided or default arguments
python3 /mnt/iusers01/mace01/t08341gt/UCF_cap_mh/src/utils/compute_save_embeddings.py  --config="$CONFIG" --save_dir="$SAVE_DIR" --name="$NAME" --model_name="$MODEL_NAME"
