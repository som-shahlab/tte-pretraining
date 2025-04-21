#!/bin/bash
#SBATCH --job-name=TotalSegmentator
#SBATCH --output=logs/output_%j.out
#SBATCH --error=logs/error_%j.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#
#SBATCH --time=48:00:00 



# Create logs directory if it doesn't exist
#SBATCH --exclude=secure-gpu-1,secure-gpu-2,secure-gpu-3
mkdir -p logs

# Execute python
python 0_rsna_pe_create_masks.py --dataset_path /share/pi/nigam/data/RSNAPE/ --output_path /share/pi/nigam/data/RSNAPE/masks --split test --limits 550 600



