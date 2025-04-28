#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1


model_choice='resnet'
nii_folder='data/inspect/anon_nii_gz' 
model_save_path='projects/zphuo/data/PE/PIXEL/model_checkpoints'  

TARGET_DIR='projects/zphuo/repos/PE_3D_multimodal/training/trash'

# finetune_labels=('12_month_readmission' 'pe_positive_nlp' '1_month_readmission' '6_month_readmission') # 
# finetune_labels=('12_month_PH' 'pe_positive_nlp' '1_month_mortality' '6_month_mortality' '12_month_mortality' '1_month_readmission'  '6_month_readmission' '12_month_readmission') # 
finetune_labels=('12_month_PH')

# label_column=('12_month_PH' 'pe_positive_nlp' '1_month_mortality' '6_month_mortality' '12_month_mortality' '1_month_readmission'  '6_month_readmission' '12_month_readmission')
label_column=('12_month_PH')

subset=''


# export WANDB_MODE=offline
export TORCH_DISTRIBUTED_DEBUG=DETAIL

module purge
# module load pytorch
# -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2

for finetune_label in "${finetune_labels[@]}"
do
    python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 1_pretrain_TTE.py \
        --model_choice $model_choice \
        --finetune_label $finetune_label \
        --label_column "${label_column[@]}" \
        --val_interval 1 \
        --label_csv 'labels_20250303.csv' \
        --max_epochs 15 \
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
        --learning_rate 1e-6 \
        --ddp \
        --dropout_prob 0.2
done


        #  --ddp \
        # --prop_val 200
        # --prop_test 150 
        # --prop_valid 3000 \
        # --test_subset
        # --linear_probe 
        # --test_subset 
        # --parquet_folder $parquet_folder \
        # --inference   
        # --prop_train 2000 \





 

