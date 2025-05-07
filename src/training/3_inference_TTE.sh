#!/bin/bash
CUDA_VISIBLE_DEVICES=0


model_choice='resnet'
nii_folder='inspect/anon_nii_gz' 
model_save_path='model_checkpoints' 

# loadmodel_path='model_checkpoints/best_metric_model_0epoch_resnet_600k_SV_0911.pth'
# loadmodel_path='model_checkpoints/best_metric_model_0epoch_resnet_600k.pth'
loadmodel_path='None'

label_column=('12_month_PH' 'pe_positive_nlp' '1_month_mortality' '6_month_mortality' '12_month_mortality' '1_month_readmission' '6_month_readmission' '12_month_readmission')  
survival_tasks=('mortality' 'readmission' 'PH' 'Atelectasis' 'Cardiomegaly' 'Consolidation' 'Edema' 'Pleural_Effusion') 

subset=''

# export WANDB_MODE=offline

# for loadmodel_path in "${loadmodel_paths[@]}"
# do 
# -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4
echo $loadmodel_path
python -m pdb \
1_pretrain_TTE.py \
--model_choice $model_choice \
--finetune_label "12_month_PH" \
--label_column "${label_column[@]}" \
--label_csv 'labels_20250303.csv' \
--max_epochs 20 \
--batch_size 1 \
--nii_folder $nii_folder \
--model_save_path $model_save_path \
--num_proc 10 \
--month_date_hour '061305' \
--from_pretrained_tokenizer \
--use_cachedataset \
--learning_rate 1e-5 \
--dropout_prob 0.3 \
--inference \
--loadmodel_path $loadmodel_path \
--survival_tasks "${survival_tasks[@]}" 




# --generate_train_features 
# --tune_linearprobe
# --prop_train 0 1000  \
# --prop_test 0 1000 
# --prop_train 0 200  \
# --pretrained_path_swinUNETR '' 
# --loadmodel_path $loadmodel_path \
# --use_crop
# --use_checkpoint \
# --linear_probe \
#  --test_subset    
# --frozen_vocab_layer \
# --use_cachedataset \
# --mixed_precision \



 

