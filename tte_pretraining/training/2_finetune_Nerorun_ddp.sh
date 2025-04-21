#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


model_choice='densenet'
nii_folder='/local-scratch-nvme/nigam/PE/anon_nii_gz/anon_nii_gz' 
model_save_path='/local-scratch/nigam/datasets/PE/model_checkpoints' 
# loadmodel_path='/local-scratch/nigam/datasets/PE/inspect/model_checkpoints/best_metric_model_34epoch_densenet_tte_0326.pth' 
# parquet_folder="/local-scratch/nigam/datasets/PE/inspect/timelines_smallfiles_meds"
TARGET_DIR='/share/pi/nigam/projects/zphuo/repos/PE_3D_multimodal/training/trash'

finetune_labels=('12_month_PH' '1_month_mortality' '6_month_mortality' '12_month_mortality') # 'pe_positive_nlp'  '1_month_readmission' '6_month_readmission' '12_month_readmission' 

label_column=('12_month_PH' 'pe_positive_nlp' '1_month_mortality' '6_month_mortality' '12_month_mortality' '1_month_readmission' '6_month_readmission' '12_month_readmission')

subset='_subset'

# export WANDB_MODE=offline

for finetune_label in "${finetune_labels[@]}"
do
    python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=6 1_pretrain_TTE.py \
        --model_choice $model_choice \
        --finetune_label $finetune_label \
        --label_column "${label_column[@]}" \
        --val_interval 1 \
        --label_csv '/share/pi/nigam/projects/zphuo/data/PE/inspect/timelines_smallfiles_meds/cohort_0.2.0_master_file_anon'$subset'.csv' \
        --max_epochs 3 \
        --vocab_size 65536 \
        --num_tasks  8192 \
        --batch_size 4 \
        --nii_folder $nii_folder \
        --model_save_path $model_save_path \
        --TARGET_DIR $TARGET_DIR \
        --use_cachedataset \
        --num_proc 10  \
        --month_date_hour '022121' \
        --from_pretrained_tokenizer \
        --learning_rate 1e-4 \
        --dropout_prob 0.1 \
        --ddp \
        --tune_linearprobe 

done

    # --tune_linearprobe \
    # --loadmodel_path $loadmodel_path \
    # --prop_train 6000 
    # --label_column '12_month_PH' \
    # --linear_probe \
    # --test_subset 
    # --parquet_folder $parquet_folder \
    # --inference   
 
    # --prop_val 2000 \  




 

