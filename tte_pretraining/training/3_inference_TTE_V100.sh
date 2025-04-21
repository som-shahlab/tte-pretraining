#!/bin/bash
CUDA_VISIBLE_DEVICES=1,2,3


model_choice='densenet'
nii_folder='/share/pi/nigam/data/inspect/anon_nii_gz' # '/local-scratch/nigam/datasets/PE/inspect/anon_nii_gz/anon_nii_gz' # '/local-scratch/nigam/datasets/PE/inspect/anon_nii_gz/anon_nii_gz'  # '/share/pi/nigam/data/inspect/crop_lung' 
model_save_path='/share/pi/nigam/projects/zphuo/data/PE/Jose_monai_MRI/model_checkpoints' # '/local-scratch/nigam/datasets/PE/model_checkpoints' 

# loadmodel_path='/share/pi/nigam/projects/zphuo/data/PE/Jose_monai_MRI/model_checkpoints/best_metric_model_0epoch_resnet_tte_600k_0710.pth' # pi/nigam/projects/zphuo/data/PE/Jose_monai_MRI/model_checkpoints/best_metric_model_4epoch_densenet_600k.pth'
loadmodel_path='/share/pi/nigam/projects/zphuo/data/PE/Jose_monai_MRI/model_checkpoints/best_metric_model_0epoch_densenet_600k.pth'
# loadmodel_path='/share/pi/nigam/projects/zphuo/data/PE/Jose_monai_MRI/model_checkpoints/best_metric_model_0epoch_densenet_600k_SV.pth'

TARGET_DIR='/share/pi/nigam/projects/zphuo/repos/PE_3D_multimodal/training/trash'
label_column=('12_month_PH' 'pe_positive_nlp' '1_month_mortality' '6_month_mortality' '12_month_mortality' '1_month_readmission' '6_month_readmission' '12_month_readmission') # 
survival_tasks=('mortality' 'readmission' 'PH' 'Atelectasis' 'Cardiomegaly' 'Consolidation' 'Edema' 'Pleural_Effusion') 

subset=''
echo $loadmodel_path

#   -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=3
python \
1_pretrain_TTE.py \
--model_choice $model_choice \
--finetune_label "12_month_PH" \
--label_column "${label_column[@]}" \
--val_interval 1 \
--label_csv '/share/pi/nigam/projects/zphuo/data/PE/inspect/timelines_smallfiles_meds/cohort_0.2.0_master_file_anon'$subset'.csv' \
--max_epochs 20 \
--vocab_size 65536 \
--num_tasks  8192 \
--TARGET_DIR $TARGET_DIR \
--batch_size 1 \
--nii_folder $nii_folder \
--model_save_path $model_save_path \
--num_proc 10 \
--month_date_hour '061305' \
--from_pretrained_tokenizer \
--use_cachedataset \
--learning_rate 1e-7 \
--dropout_prob 0.3 \
--loadmodel_path $loadmodel_path \
--survival_tasks "${survival_tasks[@]}" \
--inference 




# --generate_train_features \
# --pretrained_path_densenet ''
# --use_crop \
#  --prop_test 700 \
# --prop_train 200 \
# --prop_val 600 
# --use_checkpoint \
# --linear_probe \
# --test_subset  \
# --frozen_vocab_layer \
# --use_cachedataset \
# --mixed_precision \
# --prop_train 2000 \
# --prop_val 2000 \



 

