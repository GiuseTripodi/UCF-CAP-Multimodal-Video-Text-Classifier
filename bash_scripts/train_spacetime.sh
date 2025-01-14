#!/bin/bash --login

#$ -l v100
#$ -cwd

# Latest version of CUDA
module load libs/cuda

echo "Job is using $NGPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $NSLOTS CPU core(s)"


source /mnt/iusers01/mace01/t08341gt/env_phd/bin/activate

# Default values arg parameters
DEFAULT_DATA="/mnt/iusers01/mace01/t08341gt/UCF_cap_mh/data/UcfCap"
DEFAULT_LOG="/mnt/iusers01/mace01/t08341gt/UCF_cap_mh/data/logs"
DEFAULT_SAVE="/mnt/iusers01/mace01/t08341gt/UCF_cap_mh/data/models"
DEFAULT_NAME="test"

# Usage function to display help
usage() {
    echo "Usage: $0 [--config <config_path>] [--log_save <save_logs_path> ][--train <train_path>] [--save <save_path>] [--name <model_name>]"
    exit 1
}

# Initialize variables with default values
DATA="$DEFAULT_DATA"
LOG="$DEFAULT_LOG"
SAVE="$DEFAULT_SAVE"
NAME="$DEFAULT_NAME"

# Parse command line arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        --data)
            DATA="$2"
            shift 2
            ;;
        --log)
            LOG="$2"
            shift 2
            ;;
        --save)
            SAVE="$2"
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
python3 /mnt/iusers01/mace01/t08341gt/UCF_cap_mh/train_spacetime.py  --data="$DATA" --save="$SAVE" --log="$LOG" --name="$NAME"
