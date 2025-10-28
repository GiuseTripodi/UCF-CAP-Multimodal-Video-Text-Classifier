#!/bin/bash --login

#SBATCH -p gpuA              # A100 (80GB) GPUs  [up to 12 CPU cores per GPU permitted]
### Required flags
#SBATCH -G 1                 # (or --gpus=N) Number of GPUs
#SBATCH -t 1-0               # Wallclock timelimit (1-0 is one day, 4-0 is max permitted)

# Latest version of CUDA
module load libs/cuda
echo "Job is using $NGPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $NSLOTS CPU core(s)"
source /mnt/iusers01/mace01/t08341gt/env_phd/bin/activate

# Default values arg parameters
DEFAULT_CONFIG="/mnt/iusers01/mace01/t08341gt/UCF_cap_mh/configs/ucf-cap.json"
DEFAULT_SAVE_DIR="/mnt/iusers01/mace01/t08341gt/UCF_cap_mh/data"
DEFAULT_NAME="TESTATRICES"
DEFAULT_MODEL_NAME="space_time_Test_interpolation_25-10-25"

# Usage function to display help
usage() {
    echo "Usage: $0 [--config <config_path>] [--model_name <model_name> ] [--save_dir <save_dir>] [--name <name>]"
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
        --model_name)
            MODEL_NAME="$2"
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
python3 /mnt/iusers01/mace01/t08341gt/UCF_cap_mh/src/space_time_transformer/eval.py  --model_name="$MODEL_NAME" --save_dir="$SAVE_DIR" --config="$CONFIG" --name="$NAME"
