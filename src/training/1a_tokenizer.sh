#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


model_choice='densenet_tte'
nii_folder='data/inspect/anon_nii_gz'  
model_save_path='projects/zphuo/data/PE/PIXEL/model_checkpoints/'    
TARGET_DIR='training/trash'
label_csv='projects/zphuo/data/PE/inspect/note$ Final_labels_20250303.csv'
subset=''
ontology_path='projects/zphuo/data/PE/inspect/inspect_ontology.pkl'

python 1_pretrain_TTE.py \
    --model_choice $model_choice \
    --label_column '12_month_PH' \
    --val_interval 1 \
    --label_csv $label_csv \
    --max_epochs 10 \
    --vocab_size 65536 \
    --num_tasks 8192 \
    --TARGET_DIR $TARGET_DIR \
    --batch_size 1 \
    --nii_folder $nii_folder \
    --model_save_path $model_save_path \
    --num_proc 32 \
    --final_layer_size 1024 \
    --only_train_tokenizer 




 

