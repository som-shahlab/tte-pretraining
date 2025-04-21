#!/bin/bash

# Define variables
total_iterations=25000
increment=1000

# Execute python in a loop
for ((i = 0; i <= total_iterations; i += increment)); do
    start=$((i))
    end=$((i + increment))
    cmd="python 1_inspect_create_masks.py --dataset_path /share/pi/nigam/data/inspect/anon_nii_gz --output_path /share/pi/nigam/data/inspect/TS_crop --start $start --end $end"

    # SBATCH directives inside the loop
    sbatch <<< \
"#!/bin/bash
#SBATCH --job-name=InspectTS_${start}_${end}
#SBATCH --output=logs/output_${start}_${end}_%j.out
#SBATCH --error=logs/error_${start}_${end}_%j.err
#SBATCH --mem=100gb
#SBATCH -c 5
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
echo \"$cmd\"
# Uncomment below to actually run the command
eval \"$cmd\"
"

           
done