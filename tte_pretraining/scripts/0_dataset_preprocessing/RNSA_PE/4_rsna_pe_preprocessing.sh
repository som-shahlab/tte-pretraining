#!/bin/bash
#SBATCH --job-name=preprocessdataset
#SBATCH --output=../logs/output_%j.out
#SBATCH --error=../logs/error_%j.err
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --mem=350G
#SBATCH --cpus-per-task=20

#SBATCH --time=48:00:00


# Create logs directory if it doesn't exist
mkdir -p logs

# Execute python
python 1_rsna_pe_dicom_2_nifti.py --dataset_path /share/pi/nigam/data/RSNAPE/ --split train
#python 1_rsna_pe_dicom_2_nifti.py --dataset_path /share/pi/nigam/data/RSNAPE/ --split test



