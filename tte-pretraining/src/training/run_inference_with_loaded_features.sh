#!/bin/bash

# Default values
DATA_DIR="./data/features"
MODEL_SAVE_PATH="./models/linear_probe"
MODEL_CHOICE="densenet_600k_crop"
EPOCH=-1
MONTH_DATE=$(date +%m%d)
FEATURE_PREFIX=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --model_save_path)
      MODEL_SAVE_PATH="$2"
      shift 2
      ;;
    --model_choice)
      MODEL_CHOICE="$2"
      shift 2
      ;;
    --epoch)
      EPOCH="$2"
      shift 2
      ;;
    --month_date)
      MONTH_DATE="$2"
      shift 2
      ;;
    --tune_linearprobe)
      TUNE_LINEARPROBE="--tune_linearprobe"
      shift
      ;;
    --feature_prefix)
      FEATURE_PREFIX="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Create directories if they don't exist
mkdir -p "$MODEL_SAVE_PATH"

# Run the Python script
python 3b_load_features_inference.py \
  --data_dir "$DATA_DIR" \
  --model_save_path "$MODEL_SAVE_PATH" \
  --model_choice "$MODEL_CHOICE" \
  --epoch "$EPOCH" \
  --month_date "$MONTH_DATE" \
  ${TUNE_LINEARPROBE:+"$TUNE_LINEARPROBE"} \
  ${FEATURE_PREFIX:+"--feature_prefix" "$FEATURE_PREFIX"}
