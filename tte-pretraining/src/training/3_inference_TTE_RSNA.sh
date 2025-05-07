#!/bin/bash
CUDA_VISIBLE_DEVICES=0

model_choice='unet'
nii_folder='RSNAPE/nifti/train' # 'anon_nii_gz/anon_nii_gz'  # 'inspect/crop_lung' 
model_save_path='model_checkpoints' # 'model_checkpoints' 


loadmodel_path='model_checkpoints/best_metric_model_4epoch_unet_50k.pth'
#'model_checkpoints/best_metric_model_0epoch_unet_50k_SV_0911.pth'
#'model_checkpoints/best_metric_model_2epoch_resnet_600k.pth' 

TARGET_DIR='training/trash'
label_column=('negative_exam_for_pe' 'rv_lv_ratio_gte_1' 'chronic_pe' 'central_pe' 'leftsided_pe' 'acute_and_chronic_pe' 'rv_lv_ratio_lt_1' 'indeterminate' 'rightsided_pe')
survival_tasks=() 

subset=''
echo $model_choice
echo $loadmodel_path

#    -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=3
python \
1_pretrain_TTE.py \
--model_choice $model_choice \
--dataformat 'nii_gz_RSNAPE' \
--finetune_label "12_month_PH" \
--label_column "${label_column[@]}" \
--val_interval 1 \
--label_csv 'labels_20250303.csv' \
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
--inference 




# --tune_linearprobe
# --generate_train_features \
# --tune_linearprobe \
# --multitask \
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



 

