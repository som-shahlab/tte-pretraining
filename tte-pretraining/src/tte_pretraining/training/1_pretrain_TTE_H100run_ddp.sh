#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


model_choice='resnet_tte'
nii_folder='inspect/anon_nii_gz/anon_nii_gz' 
model_save_path='model_checkpoints'  
loadmodel_path='model_checkpoints/best_metric_model_2epoch_resnet_tte_600k_0716.pth'

TARGET_DIR='training/trash'
label_column=('12_month_PH' 'pe_positive_nlp' '1_month_mortality' '6_month_mortality' '12_month_mortality' '1_month_readmission'  '6_month_readmission' '12_month_readmission')

subset=''

WANDB__SERVICE_WAIT=300

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v 'cuda/lib64' | paste -sd ':' -)


module purge


#python 1_pretrain_TTE.py \
# srun torchrun --standalone nnodes=1 nproc_per_node=2 1_pretrain_TTE.py \

# -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2
python3  1_pretrain_TTE.py \
--model_choice $model_choice \
--label_column "${label_column[@]}" \
--val_interval 1 \
--label_csv 'labels_20250303.csv' \
--max_epochs 20 \
--vocab_size 65536 \
--num_tasks  8192 \
--batch_size 1 \
--nii_folder $nii_folder \
--model_save_path $model_save_path \
--TARGET_DIR $TARGET_DIR \
--use_cachedataset \
--num_proc 13  \
--month_date_hour '061305' \
--from_pretrained_tokenizer \
--learning_rate 1e-8 \
--dropout_prob 0.3 \
--loadmodel_path $loadmodel_path 



# --ddp \
# --wandb_run_id 'r27r721j' # unet
# --use_crop
# --prop_val 200
# --prop_test 150 
# --prop_valid 3000 \
# --test_subset
# --linear_probe 
# --test_subset 
# --parquet_folder $parquet_folder \
# --inference   
# --prop_train 2000 \





 

