from datetime import datetime
START_TIME = datetime.now()
print("Start Time: ", START_TIME)
import warnings

# Suppress specific FutureWarning from awswrangler module
warnings.filterwarnings("ignore", message="promote has been superseded by mode='default'.", category=FutureWarning, module="pyarrow")
warnings.filterwarnings("ignore", message="promote has been superseded by mode='default'.", category=FutureWarning, module="datasets")
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="The max_iter was reached which means the coef_ did not converge")
ConvergenceWarning('ignore')
from warnings import simplefilter
simplefilter("ignore", category=ConvergenceWarning)

import pyarrow
import logging
import os
import sys
import shutil
import tempfile
from torch.cuda.amp import autocast, GradScaler
import meds
from accelerate import Accelerator
import copy

# import matplotlib.pyplot as plt
import torch
# import pydicom
import torchvision
import tarfile
import glob

import argparse
import wandb
from tqdm import tqdm
import torch.nn as nn
from collections import OrderedDict
import femr.models.processor
import femr.models.tasks
import femr.models.tokenizer
from femr.models.tokenizer import FEMRTokenizer
import pickle
import datasets
import femr.index
import femr.splits

from torch.utils.tensorboard import SummaryWriter
import numpy as np
from monai.networks.nets import SwinUNETR
from sklearn.metrics import roc_auc_score
from sksurv.metrics import concordance_index_censored
import femr.models.tasks
import torchvision.transforms as transforms

import monai
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import DataLoader, ImageDataset, CacheDataset, PersistentDataset, SmartCacheDataset, PILReader
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirst,
    EnsureChannelFirstd,
    Compose,
    RandRotate90,
    RandRotate90d,
    Resize,
    Resized,
    ScaleIntensity,
    ScaleIntensityd,
)

import xgboost as xgb
import sklearn.linear_model
import sklearn.metrics
import sklearn.preprocessing
torch.autograd.set_detect_anomaly(True)


from networks import (DenseNet121, 
                      DenseNet121_TTE, 
                      SwinUNETRForClassification, 
                      SwinUNETRForClassification_TTE, 
                      I3DenseNet, 
                      resnet152, 
                      resnet152_TTE, 
                      ResNetV2_Mars
)

from network_louis import SwinClassifier, SwinClassifier_TTE

from utils import (
    load_pretrained_swinunetr,
    DicomDataset,
    TarImageDataset,
    CustomToOneChannel,
    run_analysis,
    CustomToOneChanneld,
    set_up_motor_task, 
    make_as_tensor,
    load_different_model,
    load_different_model_i3densenet,
    load_different_model_i3resnet,
    get_final_batch,
    logistic_regression,
    survival_probe,
    namestr,
    convert_3d_to_2d_weights,
    convert_3d_to_2d_weights_densenet,
    load_different_model_2D,
    is_nan,
    convert_to_list,
    EarlyStopping,
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
print_config()

# Set data directory

time_used = datetime.now() - START_TIME
print(time_used)

accelerator = Accelerator()

pin_memory = torch.cuda.is_available()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = accelerator.device
print(device)
print(torch.cuda.device_count())

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def train(
    args,
    dataformat,
    label_csv,
    CT_8192labels,
    model_save_path,
    loadmodel_path,
    model_choice,
    prop_train,
    prop_valid,
    prop_test,
    finetune_label,
    label_column,
    survival_tasks,
    val_interval,
    max_epochs,
    linear_probe,
    vocab_size,
    from_pretrained_tokenizer,
    nii_folder,
    inference,
    batch_size,
    accumulation_steps,
    use_cachedataset,
    parquet_folder,
    TARGET_DIR,
    ontology_path,
    num_proc,
    month_date_hour,
    only_train_tokenizer,
    final_layer_size,
    mixed_precision,
    use_checkpoint,
    num_tasks,
    frozen_vocab_layer,
    learning_rate,
    dropout_prob,
    test_subset,
    pretrained_path_swinUNETR,
    pretrained_path_densenet,
    pretrained_path_resnet,
    ddp,
    unet_out_channels,
    tune_linearprobe,
    use_crop,
    multitask,
    aim_hash,
    generate_train_features,
    wandb_run_id,
    wandb_group_name,
):
    
    from aim import Run
    import os
    # Initialize wandb run
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    if torch.cuda.device_count() > 1 and ddp:
        ddp = True
    else:
        ddp = False
    if wandb_group_name == 'None':
        wandb_group_name = None
    
    if ddp and not wandb_group_name:
        wandb_group_name = model_choice + "_" + datetime.now().strftime("%m_%d_%H")
    elif wandb_group_name:
        wandb_group_name = wandb_group_name
    else:
        wandb_group_name = None
    
    # wandb.require("service")
    device = accelerator.device


    if ddp:
        rank = int(torch.distributed.get_rank())
        # print(rank, type(rank), 'rank ===============')
        
        rank_ = int(str(device)[-1])
        # print(rank_, type(rank_), 'rank_ ===============')
    
        if rank != 0:
            os.environ['WANDB_MODE'] = 'dryrun'
            wandb_run_id = model_choice + "_" + datetime.now().strftime("%m_%d_%H") + "_rank_" + str(rank)
        
    config_dict = vars(args)
    # config_dict['learning_rate'] = learning_rate
    # config_dict.batch_size = batch_size
    # config_dict.dropout_prob = dropout_prob
    # config_dict.model_choice = model_choice
    # config_dict.linear_probe = linear_probe
    # config_dict.label_column = label_column
    # config_dict.fine_tune_label = finetune_label
    # config_dict.train_subset = True if '_subset' in label_csv else False
    # config_dict.test_subset = test_subset
    # config_dict.use_checkpoint = use_checkpoint
    config_dict['loadmodel_path'] = loadmodel_path.split("/")[-1] if loadmodel_path else None
    config_dict["learning_rate"] = learning_rate
    # config_dict.tune_linearprobe = tune_linearprobe
    # config_dict.use_crop = use_crop
    # config_dict.multitask = multitask
    # config_dict.pretrained_path_swinUNETR = pretrained_path_swinUNETR
    # config_dict.pretrained_path_densenet = pretrained_path_densenet
    # config_dict.inference = inference
    
    if (ddp and rank_ == 0) or (not ddp):
        if wandb_run_id:
            print("Resuming wandb run...")
            import sys
            print(sys.path)
            run_wandb = wandb.init(project="TTE", 
                                group=wandb_group_name,
                                entity="stanford_som", 
                                id=wandb_run_id, 
                                resume="allow", 
                                config=config_dict,
                                allow_val_change=True,
                                settings=wandb.Settings(
                                    init_timeout=300, 
                                    _service_wait=300),)
        # elif wandb_group_name:
        #     run_wandb = wandb.init(project="TTE", 
        #                         group=wandb_group_name,
        #                         resume=True,
        #                         allow_val_change=True,
        #                         entity="stanford_som", 
        #                         settings=wandb.Settings(
        #                             init_timeout=300, 
        #                             _service_wait=300),)
        else:
            print("Starting new wandb run...")
            run_wandb = wandb.init(project="TTE", 
                                   reinit=True,
                                group=wandb_group_name,
                                entity="stanford_som", 
                                config=config_dict,
                                settings=wandb.Settings(
                                    init_timeout=300, 
                                    _service_wait=300),)
    
        
        
        # run_wandb.config["loadmodel_path"] = loadmodel_path.split("/")[-1] if loadmodel_path else None
        # run_wandb.config["learning_rate"] = learning_rate

    # Initialize Aim run
    # run = Run(run_hash=aim_hash)
    
    print("Start Time: ", datetime.now())
    
    if loadmodel_path == 'None' or loadmodel_path == '':
        loadmodel_path = None
    
    # Log run parameters
    # run["hparams"] = {
    #     "learning_rate": learning_rate,
    #     "batch_size": batch_size,
    #     "dropout_prob": dropout_prob,
    #     "model_choice": model_choice,
    #     # "START_TIME": START_TIME,
    #     "linear_probe": linear_probe,
    #     "label_column": label_column,
    #     "fine_tune_label": finetune_label,
    #     "train_subset": True if '_subset' in label_csv else False,
    #     "test_subset": test_subset,
    #     "use_checkpoint": use_checkpoint,
    #     # "unet_out_channels": unet_out_channels,
    #     "loadmodel_path": loadmodel_path.split("/")[-1] if loadmodel_path else None,
    #     "tune_linearprobe": tune_linearprobe,
    #     "use_crop": use_crop,
    #     "multitask": multitask,
    #     "pretrained_path_swinUNETR": pretrained_path_swinUNETR,
    #     "pretrained_path_densenet": pretrained_path_densenet,
    #     "inference": inference,
    # }
    

    import pandas as pd
    
    data_types = {label_col: 'str' for label_col in label_column}
    
    if use_crop:
        spatial_size = (224, 192, 160)
        assert 'crop_lung' in nii_folder, "Please use lung_crop folder for cropping experiments"
    else:
        spatial_size = (224, 224, 224) 
        
    if 'crop_lung' in nii_folder:
        spatial_size = (224, 192, 160)
        assert use_crop, "Please set use_crop to True for lung cropping experiments"
    else:
        spatial_size = (224, 224, 224)
        
    
    if use_cachedataset:
        train_transforms = Compose(
            [LoadImaged(keys=["image"], reader="NibabelReader"), ScaleIntensityd(keys=["image"]), EnsureChannelFirstd(keys=["image"]), Resized(keys=["image"], spatial_size=spatial_size), 
            RandRotate90d(keys=["image"]), 
            CustomToOneChanneld(keys=["image"])]
        )
        val_transforms = Compose(
            [LoadImaged(keys=["image"], reader="NibabelReader"), ScaleIntensityd(keys=["image"]), EnsureChannelFirstd(keys=["image"]), Resized(keys=["image"], spatial_size=spatial_size), CustomToOneChanneld(keys=["image"])]
        )
        
    else:
        # Define transforms
        train_transforms = Compose(
            [ScaleIntensity(), EnsureChannelFirst(), Resize(spatial_size), RandRotate90(), CustomToOneChannel()]
        )

        val_transforms = Compose(
            [ScaleIntensity(), EnsureChannelFirst(), Resize(spatial_size), CustomToOneChannel()]
        )
    


    starting_epoch = 0
    
    # Define nifti dataset, data loader
    if False: 
    # dataformat == "nifti":
        check_ds = ImageDataset(
            image_files=images, labels=labels, transform=train_transforms
        )
        check_loader = DataLoader(
            check_ds, batch_size=1, num_workers=1, pin_memory=pin_memory, timeout=120
        )

        im, label = monai.utils.misc.first(check_loader)
        print(type(im), im.shape, label, label.shape)

        # create a training data loader
        train_ds = ImageDataset(
            image_files=images[:10], labels=labels[:10], transform=train_transforms
        )
        train_loader = DataLoader(
            train_ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=pin_memory
        )

        # create a validation data loader
        val_ds = ImageDataset(
            image_files=images[-10:], labels=labels[-10:], transform=val_transforms
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, num_workers=2, pin_memory=pin_memory
        )

    elif dataformat == "nii_gz":        

        # pretriaining + inference both need train labels
        if True:
            label_df = pd.read_csv(label_csv, dtype=data_types)
            label_df_tte = pd.read_csv(label_csv.replace('anon', 'anon_tte'), dtype=data_types)
            if 'same_visit_8192' in label_column:
                path = CT_8192labels
                with open(path, 'rb') as f:
                    CT_8192labels = pickle.load(f)

            labels = {}
            labels_keys_dict = {}
            image_paths = []
            
            for label_col in label_column:
                if label_col == 'same_visit_8192':
                    continue
                if label_col not in labels_keys_dict:
                    labels_keys_dict[label_col] = {}
                label_dict = label_df[label_col].value_counts().to_dict()
                keys = list(label_dict.keys())
                if 'True' in keys and 'False' in keys:
                    pos_key = 'True'
                    neg_key = 'False'
                else:
                    pos_key = keys[0]
                    neg_key = keys[1]
                labels_keys_dict[label_col] = {'pos_key': pos_key, 'neg_key': neg_key}
            
            for idx, row in label_df.iterrows():
                for label_col in label_column:
                    if label_col not in labels:
                        labels[label_col] = []
                    if label_col == 'same_visit_8192':
                        try:
                            labels[label_col].append(CT_8192labels[row['patient_datetime']])
                        except:
                            labels[label_col].append([np.nan] * 8192)
                    else:
                        if row[label_col] == 'True':
                            labels[label_col].append('1')
                        elif row[label_col] == 'False':
                            labels[label_col].append('0')
                        elif row[label_col] == 'Censored':
                            labels[label_col].append('Censored')
                image_paths.append(
                    nii_folder
                    + "/"
                    + str(row["patient_id"])
                    + "_"
                    + row["procedure_time"].replace(":", "_").replace("T", "_")
                    + ".nii.gz"
                )

            train_idx = label_df["split"] == "train"
            labels_train = {}
            for label_col in label_column:
                if label_col == 'same_visit_8192':
                    labels_train[label_col] = np.array(labels[label_col])[train_idx]
                    continue
                if label_col not in labels_train:
                    labels_train[label_col] = []
                labels_train[label_col] = np.array(labels[label_col])[train_idx]
                
            for label_col in label_column:
                if label_col == 'same_visit_8192':
                    continue
                print(label_col, np.unique(labels_train[label_col], return_counts=True), 'labels_train')
            if 'same_visit_8192' in label_column:
                labels_train_df = pd.DataFrame(labels_train['same_visit_8192'])
            else:
                labels_train_df = pd.DataFrame(labels_train)
                print(labels_train_df.columns.tolist(), 'labels_train_df')
            
            image_paths_train = np.array(image_paths)[train_idx]

            if prop_train:
                image_paths_train = image_paths_train[prop_train[0]:prop_train[1]]
                
            # full val set
            if not test_subset:
                label_csv_full = label_csv.replace('_subset', '')
            else:
                label_csv_full = label_csv
            label_df_full = pd.read_csv(label_csv_full, dtype=data_types)
            labels = {}
            image_paths = []
            for idx, row in label_df_full.iterrows():
                for label_col in label_column:
                    if label_col not in labels:
                        labels[label_col] = []
                    if label_col == 'same_visit_8192':
                        try:
                            labels[label_col].append(CT_8192labels[row['patient_datetime']])
                        except:
                            labels[label_col].append([np.nan] * 8192)
                    else:    
                        if row[label_col] == 'True':
                            labels[label_col].append('1')
                        elif row[label_col] == 'False':
                            labels[label_col].append('0')
                        elif row[label_col] == 'Censored':
                            labels[label_col].append('Censored')
                image_paths.append(
                    nii_folder
                    + "/"
                    + str(row["patient_id"])
                    + "_"
                    + row["procedure_time"].replace(":", "_").replace("T", "_")
                    + ".nii.gz"
                )
            val_idx = label_df_full["split"] == "valid"
            labels_valid = {}
            for label_col in label_column:
                if label_col == 'same_visit_8192':
                    labels_valid[label_col] = np.array(labels[label_col])[val_idx]
                    continue
                if label_col not in labels_valid:
                    labels_valid[label_col] = []
                labels_valid[label_col] = np.array(labels[label_col])[val_idx]
            for label_col in label_column:
                if label_col == 'same_visit_8192':
                    continue
                print(label_col, np.unique(labels_valid[label_col], return_counts=True), 'labels_valid')
            if 'same_visit_8192' in label_column:
                labels_valid_df = pd.DataFrame(labels_valid['same_visit_8192'])
            else:
                labels_valid_df = pd.DataFrame(labels_valid)
                print(labels_valid_df.columns.tolist(), 'labels_valid_df')
            image_paths_valid = np.array(image_paths)[val_idx]
            
            if prop_valid:
                image_paths_valid = image_paths_valid[prop_valid[0]:prop_valid[1]]
            
            print(len(image_paths_train), 'len(image_paths_train)')
            if use_cachedataset:
                
                data_train = []
                for i in range(len(image_paths_train)):
                    one_entry = {'image': image_paths_train[i], 'label': list(labels_train_df.iloc[i].values.tolist()), 'image_path': image_paths_train[i], }
                    data_train.append(one_entry) 
                data_val = []
                for i in range(len(image_paths_valid)):
                    one_entry = {'image': image_paths_valid[i], 'label': list(labels_valid_df.iloc[i].values.tolist()), 'image_path': image_paths_valid[i],}
                    data_val.append(one_entry)
                
                    
                # cache_dir maxed out        
                if '_tte' in model_choice or 'resnet_mtl' in model_choice:
                    cache_dir=os.path.join(model_save_path, 'cache_dir_tte')  
                else: 
                    cache_dir=os.path.join(model_save_path, 'cache_dir') 
                                 
                print('cache_dir---------', cache_dir)                 
                 
                # v100 issue, save to here if dir exisits
                cache_dir_v = '/local-scratch-nvme/nigam/PE/model_checkpoints/cache_dir'
                if os.path.isdir(cache_dir_v):
                    cache_dir = '/local-scratch/nigam/datasets/PE/model_checkpoints/cache_dir'
                    print('cache_dir_v---------', cache_dir_v)
                 
                train_ds = PersistentDataset(
                    data=data_train,
                    transform=train_transforms,
                    #cache_num=9223,
                    cache_dir=cache_dir,
                )
                val_ds = PersistentDataset(
                    data=data_val,
                    transform=val_transforms,
                    #cache_num=9223,
                    cache_dir=cache_dir,
                )
            
            else:
                # create a training data 
                train_ds = ImageDataset(
                    image_files=image_paths_train,
                    labels=labels_train,
                    transform=train_transforms,
                )
                
                # create a validation data 
                val_ds = ImageDataset(
                    image_files=image_paths_valid,
                    labels=labels_valid,
                    transform=val_transforms,
                )

            train_loader = DataLoader(
                    train_ds, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=pin_memory
                )
            val_loader = DataLoader(
                    val_ds, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=pin_memory
                )

        # inference only
        if inference:

            # full test loader
            if not test_subset:
                label_csv_full = label_csv.replace('_subset', '')
            else:
                label_csv_full = label_csv
            label_df_full = pd.read_csv(label_csv_full, dtype=data_types)
            label_df = pd.read_csv(label_csv, dtype=data_types)
            
            labels ={}
            labels_keys_dict = {}
            image_paths = []
            
            # label column unique values, top as 1, second top as 0
            for label_col in label_column:
                if label_col == 'same_visit_8192':
                    continue
                if label_col not in labels_keys_dict:
                    labels_keys_dict[label_col] = {}
                label_dict = label_df[label_col].value_counts().to_dict()
                keys = list(label_dict.keys())
                if 'True' in keys and 'False' in keys:
                    pos_key = 'True'
                    neg_key = 'False'
                else:
                    pos_key = keys[0]
                    neg_key = keys[1]
                labels_keys_dict[label_col] = {'pos_key': pos_key, 'neg_key': neg_key}

            
            for idx, row in label_df_full.iterrows():
                for label_col in label_column:
                    if label_col == 'same_visit_8192':
                        continue
                    if label_col not in labels:
                        labels[label_col] = []
                    if label_col == 'same_visit_8192':
                        try:
                            labels[label_col].append(CT_8192labels[row['patient_datetime']])
                        except:
                            labels[label_col].append([np.nan] * 8192)
                    else:
                        if row[label_col] == 'True':
                            labels[label_col].append('1')
                        elif row[label_col] == 'False':
                            labels[label_col].append('0')
                        elif row[label_col] == 'Censored':
                            labels[label_col].append('Censored')
                image_paths.append(
                    nii_folder
                    + "/"
                    + str(row["patient_id"])
                    + "_"
                    + row["procedure_time"].replace(":", "_").replace("T", "_")
                    + ".nii.gz"
                )

            test_idx = label_df_full["split"] == "test"
            labels_test = {}
            for label_col in label_column:
                if label_col == 'same_visit_8192':
                    labels_test[label_col] = np.array(labels[label_col])[test_idx]
                    continue
                if label_col not in labels_test:
                    labels_test[label_col] = []
                labels_test[label_col] = np.array(labels[label_col])[test_idx]
                
            for label_col in label_column:
                if label_col == 'same_visit_8192':
                    continue
                print(label_col, np.unique(labels_test[label_col], return_counts=True), 'labels_test')
            if 'same_visit_8192' in label_column:
                labels_test_df = pd.DataFrame(labels_test['same_visit_8192'])
            else:
                labels_test_df = pd.DataFrame(labels_test)
                print(labels_test_df.columns.tolist(), 'labels_test_df')
                
            image_paths_test = np.array(image_paths)[test_idx]
            
            if prop_test:
                image_paths_test = image_paths_test[prop_test[0]:prop_test[1]]
                        
            # generate test features 
            if True:
                if use_cachedataset:
                    data_test = []
                    for i in range(len(image_paths_test)):
                        one_entry = {'image': image_paths_test[i], 'label': list(labels_test_df.iloc[i].values.tolist()), 'image_path': image_paths_test[i], }
                        data_test.append(one_entry)
                    test_ds = PersistentDataset(
                        data=data_test,
                        transform=val_transforms,
                        #cache_num=9223,
                        cache_dir=cache_dir,
                    )
                else:
                    test_ds = ImageDataset(
                        image_files=image_paths_test,
                        labels=labels_test,
                        transform=val_transforms,
                    )
                test_loader = DataLoader(
                    test_ds, batch_size=batch_size, num_workers=3, shuffle=True, pin_memory=pin_memory
                )
                print(test_loader.num_workers, 'num workers test' )
    
    elif dataformat == "nii_gz_RSNAPE":
        
        label_df_tte = None
        
        label_df = pd.read_csv(label_csv, dtype=data_types)


        image_paths = []
        labels = []
        for idx, row in tqdm(label_df.iterrows(), total = len(label_df)):
            labels.append(row[label_column].values.tolist())
            image_paths.append(
                nii_folder
                + "/"
                + str(row["study_series"])
                + '/'
                + str(row["nii_name"])
            )
            
        # train/val/test 80/5/15
        indices = np.arange(len(image_paths))
        np.random.seed(0)
        np.random.shuffle(indices)
        train_indices = indices[:int(0.8*len(indices))]
        val_indices = indices[int(0.8*len(indices)):int(0.85*len(indices))]
        test_indices = indices[int(0.85*len(indices)):]

        image_paths_train = np.array(image_paths)[train_indices]
        labels_train = np.array(labels)[train_indices]
        image_paths_valid = np.array(image_paths)[val_indices]
        labels_valid = np.array(labels)[val_indices]
        image_paths_test = np.array(image_paths)[test_indices]
        labels_test = np.array(labels)[test_indices]

        if prop_train:
            image_paths_train = image_paths_train[prop_train[0]:prop_train[1]]
        if prop_valid:
                image_paths_valid = image_paths_valid[prop_valid[0]:prop_valid[1]]
        if prop_test:
            image_paths_test = image_paths_test[prop_test[0]:prop_test[1]]
        print(len(image_paths_train), '== len(image_paths_train)', 
              len(image_paths_valid), '== len(image_paths_valid)', 
              len(image_paths_test), '== len(image_paths_test)')


        image_paths_train_ = image_paths_train
        image_paths_valid_ = image_paths_valid
        image_paths_test_ = image_paths_test
                
        if use_cachedataset:
            data_train = []
            for i in range(len(image_paths_train)):
                one_entry = {'image': str(image_paths_train[i]), 'label': str(labels_train[i])} #, 'image_path': [str(image_paths_train[i])]}
                data_train.append(one_entry)
            data_val = []
            for i in range(len(image_paths_valid)):
                one_entry = {'image': str(image_paths_valid[i]), 'label': str(labels_valid[i])} #, 'image_path': [str(image_paths_valid[i])]}
                data_val.append(one_entry)
            data_test = []
            for i in range(len(image_paths_test)):
                one_entry = {'image': str(image_paths_test[i]), 'label': str(labels_test[i])} #, 'image_path': [str(image_paths_test[i])]}
                data_test.append(one_entry)
                
            train_transforms = Compose(
                [LoadImaged(keys=["image"], reader="NibabelReader"), ScaleIntensityd(keys=["image"]), EnsureChannelFirstd(keys=["image"]), Resized(keys=["image"], spatial_size=spatial_size), 
                RandRotate90d(keys=["image"]), 
                CustomToOneChanneld(keys=["image"])]
            )
            val_transforms = Compose(
                [LoadImaged(keys=["image"], reader="NibabelReader"), ScaleIntensityd(keys=["image"]), EnsureChannelFirstd(keys=["image"]), Resized(keys=["image"], spatial_size=spatial_size), CustomToOneChanneld(keys=["image"])]
            )    
                
            # v100 node cache_dir maxed out
            if '_tte' in model_choice:
                cache_dir=os.path.join(model_save_path, 'cache_dir_tte')  
            else: 
                cache_dir=os.path.join(model_save_path, 'cache_dir')
                
            # save to here if dir exisits
            cache_dir_v = '/local-scratch-nvme/nigam/PE/model_checkpoints/cache_dir'
            if os.path.isdir(cache_dir_v):
                cache_dir = '/local-scratch/nigam/datasets/PE/inspect/model_checkpoints/cache_dir/'
                
            # GPU partition a100 cache_dir
            cache_dir_a = '/local-scratch/nigam/users/zphuo'
            if os.path.isdir(cache_dir_a):
                if loadmodel_path:
                    if 'unet_50k' in loadmodel_path:
                        cache_dir = os.path.join(cache_dir_a, 'cache_dir_tte')
                    if 'resnet' in model_choice:
                        cache_dir = os.path.join(cache_dir_a, 'cache_dir', 'cache_dir')
                    else:
                        cache_dir = os.path.join(cache_dir_a, 'cache_dir')
                    print('cache_dir_a---------', cache_dir_a)      
                
            train_ds = PersistentDataset(
                data=data_train,
                transform=train_transforms,
                #cache_num=9223,
                cache_dir=cache_dir,
            )
            val_ds = PersistentDataset(
                data=data_val,
                transform=val_transforms,
                #cache_num=9223,
                cache_dir=cache_dir,
            )
            test_ds = PersistentDataset(
                data=data_test,
                transform=val_transforms,
                #cache_num=9223,
                cache_dir=cache_dir,
            )
            
            # DON'T SHUFFLE. image_path is read sequentially
            train_loader = DataLoader(
                    train_ds, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=pin_memory)    
            val_loader = DataLoader(
                    val_ds, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=pin_memory)
            test_loader = DataLoader(
                    test_ds, batch_size=batch_size, num_workers=3, shuffle=False, pin_memory=pin_memory)
    
    elif dataformat == 'jpg_chexpert':
        label_df_tte = None
        
        label_df = pd.read_csv(label_csv, dtype=data_types)
        
        if prop_train:
            label_df = label_df.iloc[prop_train[0]:prop_train[1], :]
    
        image_paths = []
        labels = []
        for idx, row in tqdm(label_df.iterrows(), total = len(label_df)):
            
            row_label = row[label_column].values.tolist()
            
            # handle chexpert undertain labels, -1 -> 1
            value_mapping = {-1: '1', 0: '0', 1: '1', 0.0: '0', 1.0: '1', -1.0: '1'}
            new_list = [0 if is_nan(x) else value_mapping.get(x, x) for x in row_label]
            labels.append(new_list)
            image_paths.append(
                nii_folder
                + "/"
                + str(row["Path"])
            )
        
        label_df_test = pd.read_csv(label_csv_test, dtype=data_types)
    
        for idx, row in tqdm(label_df_test.iterrows(), total = len(label_df_test)):
            
            row_label = row[label_column].values.tolist()
            
            # handle chexpert undertain labels, -1 -> 1
            value_mapping = {-1: '1', 0: '0', 1: '1', 0.0: '0', 1.0: '1', -1.0: '1'}
            new_list = [0 if is_nan(x) else value_mapping.get(x, x) for x in row_label]
            labels.append(new_list)
            image_paths.append(
                nii_folder
                + "/"
                + str(row["Path"])
            )
            
        indices = np.arange(len(image_paths))
        np.random.seed(42)
        np.random.shuffle(indices)
        train_indices = indices[:int(0.8*len(indices))]
        val_indices = indices[int(0.8*len(indices)):int(0.85*len(indices))]
        test_indices = indices[int(0.85*len(indices)):]
        
        image_paths_train = np.array(image_paths)[train_indices]
        labels_train = np.array(labels)[train_indices]
        image_paths_valid = np.array(image_paths)[val_indices]
        labels_valid = np.array(labels)[val_indices]
        image_paths_test = np.array(image_paths)[test_indices]
        labels_test = np.array(labels)[test_indices]
        image_paths_train_ = image_paths_train
        image_paths_valid_ = image_paths_valid
        image_paths_test_ = image_paths_test
            
        if use_cachedataset:
            data_train = []
            for i in range(len(image_paths_train)):
                one_entry = {'image': str(image_paths_train[i]), 'label': str(labels_train[i])} #, 'image_path': [str(image_paths_train[i])]}
                data_train.append(one_entry)
            data_val = []
            for i in range(len(image_paths_valid)):
                one_entry = {'image': str(image_paths_valid[i]), 'label': str(labels_valid[i])} #, 'image_path': [str(image_paths_valid[i])]}
                data_val.append(one_entry)
            data_test = []
            for i in range(len(image_paths_test)):
                one_entry = {'image': str(image_paths_test[i]), 'label': str(labels_test[i])} #, 'image_path': [str(image_paths_test[i])]}
                data_test.append(one_entry)
                
            train_transforms = Compose(
                [LoadImaged(keys=["image"], reader=PILReader()), ScaleIntensityd(keys=["image"]), EnsureChannelFirstd(keys=["image"]), Resized(keys=["image"], spatial_size=(224, 224)), 
                RandRotate90d(keys=["image"]), 
                CustomToOneChanneld(keys=["image"])]
            )
            val_transforms = Compose(
                [LoadImaged(keys=["image"], reader=PILReader()), ScaleIntensityd(keys=["image"]), EnsureChannelFirstd(keys=["image"]), Resized(keys=["image"], spatial_size=(224, 224)), CustomToOneChanneld(keys=["image"])]
            )    
            
            # v100 node cache_dir maxed out
            if '_tte' in model_choice:
                cache_dir=os.path.join(model_save_path, 'cache_dir_tte')  
            else: 
                cache_dir=os.path.join(model_save_path, 'cache_dir')
                
            # save to here if dir exisits
            cache_dir_v = '/local-scratch-nvme/nigam/PE/model_checkpoints/cache_dir'
            if os.path.isdir(cache_dir_v):
                cache_dir = '/local-scratch/nigam/datasets/PE/inspect/model_checkpoints/cache_dir/'
                
            # GPU partition a100 cache_dir
            cache_dir_a = '/local-scratch/nigam/users/zphuo'
            if os.path.isdir(cache_dir_a):
                if loadmodel_path:
                    if 'unet_50k' in loadmodel_path:
                        cache_dir = os.path.join(cache_dir_a, 'cache_dir_tte')
                    if 'resnet' in model_choice:
                        cache_dir = os.path.join(cache_dir_a, 'cache_dir', 'cache_dir')
                    else:
                        cache_dir = os.path.join(cache_dir_a, 'cache_dir')
                    print('cache_dir_a---------', cache_dir_a)   
            
            train_ds = PersistentDataset(
                data=data_train,
                transform=train_transforms,
                #cache_num=9223,
                cache_dir=cache_dir,
            )
            val_ds = PersistentDataset(
                data=data_val,
                transform=val_transforms,
                #cache_num=9223,
                cache_dir=cache_dir,
            )
            test_ds = PersistentDataset(
                data=data_test,
                transform=val_transforms,
                #cache_num=9223,
                cache_dir=cache_dir,
            )    
            
            # DON'T SHUFFLE. image_path is read sequentially
            train_loader = DataLoader(
                    train_ds, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=pin_memory, drop_last=True)    
            val_loader = DataLoader(
                    val_ds, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=pin_memory, drop_last=True)
            test_loader = DataLoader(
                    test_ds, batch_size=batch_size, num_workers=3, shuffle=False, pin_memory=pin_memory, drop_last=True)
                
    elif False:
    # dataformat == "dicom":
        tar_folder = "/local-scratch/nigam/datasets/PE/inspect/anon_dicoms_tar"

        label_df = pd.read_csv(label_csv)
        label_column = "pe_positive_nlp"
        labels = label_df[label_column].values
        labels = [1 if label == True else 0 for label in labels]
        tar_files = [
            os.path.join(tar_folder, str(patient_id) + ".tar")
            for patient_id in label_df["patient_id"].values
        ]

        train_idx = label_df["split"] == "train"
        valid_idx = label_df["split"] == "valid"
        test_idx = label_df["split"] == "test"

        labels_train = np.array(labels)[train_idx]
        labels_valid = np.array(labels)[valid_idx]
        labels_test = np.array(labels)[test_idx]

        tar_files_train = np.array(tar_files)[train_idx]
        tar_files_valid = np.array(tar_files)[valid_idx]
        tar_files_test = np.array(tar_files)[test_idx]

        check_ds = TarImageDataset(
            tar_files=tar_files, labels=labels, transform=train_transforms
        )
        for tar in tar_files:
            if type(tar) != str:
                print(tar)
        check_loader = DataLoader(
            check_ds, batch_size=3, num_workers=2, pin_memory=pin_memory
        )

        im, label = monai.utils.misc.first(check_loader)
        print(type(im), im.shape, label, label.shape)

        # create a training data loader
        train_ds = TarImageDataset(
            tar_files=tar_files_train, labels=labels_train, transform=train_transforms
        )
        train_loader = DataLoader(
            train_ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=pin_memory
        )

        # create a validation data loader
        val_ds = TarImageDataset(
            image_files=tar_files_valid, labels=labels_valid, transform=val_transforms
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, num_workers=2, pin_memory=pin_memory
        )
    
    else:
        print("Data format not supported!! ------------------")
    
    
    # selecting model
    if model_choice == "unet" or model_choice == "unet_mtl":
        motor_task, tokenizer, train_dataset, valid_dataset, test_dataset, processor, index_train, index_valid, index_test, num_tasks, ontology = set_up_motor_task(TARGET_DIR, from_pretrained_tokenizer, month_date_hour, num_proc, label_csv, ontology_path, inference, vocab_size, START_TIME, parquet_folder, final_layer_size, num_tasks, test_subset)
        # Define your SwinUNETR parameters in a dictionary
        swin_unetr_params = {
            "img_size": spatial_size,
            "in_channels": 1,
            "out_channels": unet_out_channels,  # Used for segmentation, but will be adapted
            "feature_size": 48,
            "drop_rate": 0.0,
            "attn_drop_rate": 0.0,
            "dropout_path_rate": 0.0,
            "use_checkpoint": True,
        }

        if not multitask:
            num_classes = 2
        else:
            if 'same_visit_8192' in label_column:
                num_classes = 8192
            else:
                num_classes = len(label_column)

        # Initialize the SwinUNETRForClassification model
        model = SwinUNETRForClassification(
            swin_unetr_params,
            num_classes=num_classes,  # Specify the number of classes for classification
        ).to(device)
            
        if loadmodel_path:
            state_dict = torch.load(loadmodel_path, map_location=f'cuda:{torch.cuda.current_device()}')
            # model.load_state_dict(state_dict)
            
            try:
                model = load_different_model(model, state_dict, ddp)
            except:
                state_dict = {'swin_unetr.'+k: v for k, v in state_dict.items()}
                model = load_different_model(model, state_dict, ddp)
            path_ls = loadmodel_path.split('_')
            for chunk in path_ls:
                if 'epoch' in chunk:
                    starting_epoch = int(chunk.replace('epoch', ''))  
                    break
                
        elif (not loadmodel_path) and pretrained_path_swinUNETR:
            pretrained_path = pretrained_path_swinUNETR
            model = load_pretrained_swinunetr(
                model, use_pretrained=True, pretrained_path=pretrained_path
            )
            starting_epoch = -1
            
        if pretrained_path_swinUNETR:
            model_choice += "_50k"
            
        if 'same_visit_8192' in label_column:
            model_choice += "_SV"
            
        if linear_probe:
            print("Linear probe for SwinUNETR...")
            # Freeze all layers in the model
            for name, param in model.named_parameters():
                if 'swin_unetr' in name and "out" not in name:
                    param.requires_grad = False
        
        if ddp:           
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
            
        if dataformat == "nii_gz_RSNAPE":
            model_choice += "_RSNAPE"    
            
        if use_crop:
            model_choice += "_crop"
           
    elif model_choice == "unet_tte":
        motor_task, tokenizer, train_dataset, valid_dataset, test_dataset, processor, index_train, index_valid, index_test, num_tasks, ontology = set_up_motor_task(TARGET_DIR, from_pretrained_tokenizer, month_date_hour, num_proc, label_csv, ontology_path, inference, vocab_size, START_TIME, parquet_folder, final_layer_size, num_tasks, test_subset)
        if only_train_tokenizer:
            exit()
        print('number of tasks: ', len(motor_task.pretraining_task_info))
        
        # Define your SwinUNETR parameters in a dictionary
        swin_unetr_params = {
            "img_size": spatial_size,
            "in_channels": 1,
            "out_channels": unet_out_channels,  # Used for segmentation, but will be adapted
            "feature_size": 48,
            "drop_rate": 0.0,
            "attn_drop_rate": 0.0,
            "dropout_path_rate": 0.0,
            "use_checkpoint": True,
        }

        # Initialize the SwinUNETRForClassification model
        model = SwinUNETRForClassification_TTE(
            swin_unetr_params,
            num_classes=2,  
            final_layer_size=motor_task.final_layer_size,
            time_bins=motor_task.time_bins,
            pretraining_task_info=motor_task.get_task_config().task_kwargs['pretraining_task_info'], 
            device=device, # Specify the number of classes for classification
        ).to(device)
            
        if loadmodel_path:   
            state_dict = torch.load(loadmodel_path, map_location=f'cuda:{torch.cuda.current_device()}') 
            try:
                model = load_different_model(model, state_dict, ddp)
            except:
                del state_dict['swin_unetr.out.conv.conv.weight']
                del state_dict['swin_unetr.out.conv.conv.bias']
                del state_dict['final_layer.weight']
                del state_dict['final_layer.bias']
                model = load_different_model(model, state_dict, ddp)
            path_ls = loadmodel_path.split('_')
            for chunk in path_ls:
                if 'epoch' in chunk:
                    starting_epoch = int(chunk.replace('epoch', ''))  
                    break
                
        elif (not loadmodel_path) and pretrained_path_swinUNETR:
            pretrained_path = pretrained_path_swinUNETR
            model = load_pretrained_swinunetr(
                model, use_pretrained=True, pretrained_path=pretrained_path
            )   
            
        if pretrained_path_swinUNETR:
            model_choice += "_50k" 
            
        if 'same_visit_8192' in label_column:
            model_choice += "_SV"
            
        if linear_probe:
            print("Linear probe for SwinUNETR...")
            for name, param in model.named_parameters():
                if 'swin_unetr' in name and "out" not in name:
                    param.requires_grad = False

        if ddp:
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
            
        if dataformat == "nii_gz_RSNAPE":
            model_choice += "_RSNAPE"  
            
        if use_crop:
            model_choice += "_crop"
            
    elif model_choice == "densenet" or model_choice == "densenet_mtl":
        motor_task, tokenizer, train_dataset, valid_dataset, test_dataset, processor, index_train, index_valid, index_test, num_tasks, ontology = set_up_motor_task(TARGET_DIR, from_pretrained_tokenizer, month_date_hour, num_proc, label_csv, ontology_path, inference, vocab_size, START_TIME, parquet_folder, final_layer_size, num_tasks, test_subset)
        if not multitask:
            num_classes = 2
        else:
            if 'same_visit_8192' in label_column:
                num_classes = 8192
            else:
                num_classes = len(label_column)
            
        model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=num_classes).to(device)
        if loadmodel_path:
            state_dict = torch.load(loadmodel_path, map_location=f'cuda:{torch.cuda.current_device()}')
            # model.load_state_dict(state_dict)
            model = load_different_model(model, state_dict, ddp)
            path_ls = loadmodel_path.split('_')
            for chunk in path_ls:
                if 'epoch' in chunk:
                    starting_epoch = int(chunk.replace('epoch', ''))  
                    break
        
        # Load the pretrained weights
        elif (not loadmodel_path) and pretrained_path_densenet:
            
            state_dict = torch.load(pretrained_path_densenet, map_location=f'cuda:{torch.cuda.current_device()}') 
            model = load_different_model_i3densenet(model, state_dict, ddp)
            
            starting_epoch = -1
          
        if linear_probe:
            print("Linear probe for densenet...")
            # Freeze all layers in the model
            for name, param in model.named_parameters():
                if "features" in name:
                    param.requires_grad = False
        
        if pretrained_path_densenet:
            model_choice += "_600k"
        
        if 'same_visit_8192' in label_column:
            model_choice += "_SV"
        
        if ddp:
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
            
        if dataformat == "nii_gz_RSNAPE":
            model_choice += "_RSNAPE"  
            
        if use_crop:
            model_choice += "_crop"
            
    elif model_choice == "densenet_tte":
        motor_task, tokenizer, train_dataset, valid_dataset, test_dataset, processor, index_train, index_valid, index_test, num_tasks, ontology = set_up_motor_task(TARGET_DIR, from_pretrained_tokenizer, month_date_hour, num_proc, label_csv, ontology_path, inference, vocab_size, START_TIME, parquet_folder, final_layer_size, num_tasks, test_subset)
        if only_train_tokenizer:
            exit()
        
        print('number of tasks: ', len(motor_task.pretraining_task_info))
        model = DenseNet121_TTE(spatial_dims=3, in_channels=1, out_channels=2, time_bins=motor_task.time_bins, pretraining_task_info=motor_task.get_task_config().task_kwargs['pretraining_task_info'], final_layer_size=motor_task.final_layer_size, vocab_size=tokenizer.vocab_size, device=device, use_checkpoint=use_checkpoint, dropout_prob=dropout_prob).to(device)
        
        if loadmodel_path:
            state_dict = torch.load(loadmodel_path, map_location=f'cuda:{torch.cuda.current_device()}') 
            # model.load_state_dict(state_dict)
            
            model = load_different_model(model, state_dict, ddp)
            path_ls = loadmodel_path.split('_')
            for chunk in path_ls:
                if 'epoch' in chunk:
                    starting_epoch = int(chunk.replace('epoch', ''))  
                    break
            
        elif (not loadmodel_path) and pretrained_path_densenet:
            state_dict = torch.load(pretrained_path_densenet, map_location=f'cuda:{torch.cuda.current_device()}') 
            model = load_different_model_i3densenet(model, state_dict, ddp)
            starting_epoch = -1

        if pretrained_path_densenet:
            model_choice += "_600k"
            
        if 'same_visit_8192' in label_column:
            model_choice += "_SV"
            
        # test frozen vocab layer
        if frozen_vocab_layer:
            print("Freezing vocab layer...")
            for name, param in model.named_parameters():
                if "after_class_layers" in name:
                    param.requires_grad = False
                    
        if linear_probe:
            print("Linear probe for densenet TTE...")
            # Freeze all layers in the model
            for name, param in model.named_parameters():
                if "features" in name:
                    param.requires_grad = False
                    
        if ddp:
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
            
        if dataformat == "nii_gz_RSNAPE":
            model_choice += "_RSNAPE"  
            
        if use_crop:
            model_choice += "_crop"
            
            
    elif model_choice == "resnet" or model_choice == "resnet_mtl":
        motor_task, tokenizer, train_dataset, valid_dataset, test_dataset, processor, index_train, index_valid, index_test, num_tasks, ontology = set_up_motor_task(TARGET_DIR, from_pretrained_tokenizer, month_date_hour, num_proc, label_csv, ontology_path, inference, vocab_size, START_TIME, parquet_folder, final_layer_size, num_tasks, test_subset)
        if not multitask:
            num_classes = 2
        else:
            if 'same_visit_8192' in label_column:
                num_classes = 8192
            else:
                num_classes = len(label_column)
            
        model = resnet152(n_input_channels=1, num_classes=num_classes).to(device)
        if loadmodel_path:
            state_dict = torch.load(loadmodel_path, map_location=f'cuda:{torch.cuda.current_device()}')
            model = load_different_model(model, state_dict, ddp)
            path_ls = loadmodel_path.split('_')
            for chunk in path_ls:
                if 'epoch' in chunk:
                    starting_epoch = int(chunk.replace('epoch', ''))  
                    break
        
        # Load the pretrained weights
        elif (not loadmodel_path) and pretrained_path_resnet:
            
            state_dict = torch.load(pretrained_path_resnet, map_location=f'cuda:{torch.cuda.current_device()}') 
            model = load_different_model_i3resnet(model, state_dict, ddp)
            
            starting_epoch = -1
          
        if linear_probe:
            print("Linear probe for resnet...")
            # Freeze all layers in the model
            for name, param in model.named_parameters():
                if "features" in name:
                    param.requires_grad = False
        
        if pretrained_path_resnet:
            model_choice += "_600k"
            
        if 'same_visit_8192' in label_column:
            model_choice += "_SV"
        
        if ddp:
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
            
        if dataformat == "nii_gz_RSNAPE":
            model_choice += "_RSNAPE"  
            
        if use_crop:
            model_choice += "_crop"
            
    elif model_choice == "resnet_tte":
        motor_task, tokenizer, train_dataset, valid_dataset, test_dataset, processor, index_train, index_valid, index_test, num_tasks, ontology = set_up_motor_task(TARGET_DIR, from_pretrained_tokenizer, month_date_hour, num_proc, label_csv, ontology_path, inference, vocab_size, START_TIME, parquet_folder, final_layer_size, num_tasks, test_subset)
        if only_train_tokenizer:
            exit()
        
        print('number of tasks: ', len(motor_task.pretraining_task_info))
        model = resnet152_TTE(n_input_channels=1, time_bins=motor_task.time_bins, pretraining_task_info=motor_task.get_task_config().task_kwargs['pretraining_task_info'], final_layer_size=motor_task.final_layer_size, device=device).to(device)
        
        if loadmodel_path:
            state_dict = torch.load(loadmodel_path, map_location=f'cuda:{torch.cuda.current_device()}') 
            # model.load_state_dict(state_dict)
            
            model = load_different_model(model, state_dict, ddp)
            path_ls = loadmodel_path.split('_')
            for chunk in path_ls:
                if 'epoch' in chunk:
                    starting_epoch = int(chunk.replace('epoch', ''))  
                    break
            
        elif (not loadmodel_path) and pretrained_path_resnet:
            state_dict = torch.load(pretrained_path_resnet, map_location=f'cuda:{torch.cuda.current_device()}') 
            model = load_different_model_i3resnet(model, state_dict, ddp)
            starting_epoch = -1

        if pretrained_path_resnet:
            model_choice += "_600k"
            
        # test frozen vocab layer
        if frozen_vocab_layer:
            print("Freezing vocab layer...")
            for name, param in model.named_parameters():
                if "after_class_layers" in name:
                    param.requires_grad = False
                    
        if linear_probe:
            print("Linear probe for densenet TTE...")
            # Freeze all layers in the model
            for name, param in model.named_parameters():
                if "features" in name:
                    param.requires_grad = False
                    
        if ddp:
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
            
        if dataformat == "nii_gz_RSNAPE":
            model_choice += "_RSNAPE"  
            
        if use_crop:
            model_choice += "_crop"
            
    
    elif model_choice == "resnet_tte_chexpert":
        motor_task, tokenizer, train_dataset, valid_dataset, test_dataset, processor, index_train, index_valid, index_test, num_tasks, ontology = set_up_motor_task(TARGET_DIR, from_pretrained_tokenizer, month_date_hour, num_proc, label_csv, ontology_path, inference, vocab_size, START_TIME, parquet_folder, final_layer_size, num_tasks, test_subset)
        if not multitask:
            num_classes = 2
        else:
            if 'same_visit_8192' in label_column:
                num_classes = 8192
            else:
                num_classes = len(label_column)
        model = resnet152(n_input_channels=1, spatial_dims=2).to(device)
        
        if loadmodel_path:
            state_dict = torch.load(loadmodel_path, map_location=f'cuda:{torch.cuda.current_device()}')
            state_dict = convert_3d_to_2d_weights(state_dict, model) 
            model = load_different_model_i3resnet(model, state_dict, ddp)
            
            path_ls = loadmodel_path.split('_')
            for chunk in path_ls:
                if 'epoch' in chunk:
                    starting_epoch = int(chunk.replace('epoch', ''))  
                    break
                
        # Load the pretrained weights
        elif (not loadmodel_path) and pretrained_path_resnet:
            
            state_dict = torch.load(pretrained_path_resnet, map_location=f'cuda:{torch.cuda.current_device()}') 
            state_dict = convert_3d_to_2d_weights(state_dict, model)
            model = load_different_model_i3resnet(model, state_dict, ddp)
            
            starting_epoch = -1
            
        if linear_probe:
            print("Linear probe for resnet...")
            # Freeze all layers in the model
            for name, param in model.named_parameters():
                if "features" in name:
                    param.requires_grad = False
                    
        if pretrained_path_resnet:
            model_choice += "_600k"
        
        if ddp:
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
            
        if dataformat == "nii_gz_RSNAPE":
            model_choice += "_RSNAPE"  
            
        if use_crop:
            model_choice += "_crop"
            
    elif model_choice == "densenet_tte_chexpert":
        motor_task, tokenizer, train_dataset, valid_dataset, test_dataset, processor, index_train, index_valid, index_test, num_tasks, ontology = set_up_motor_task(TARGET_DIR, from_pretrained_tokenizer, month_date_hour, num_proc, label_csv, ontology_path, inference, vocab_size, START_TIME, parquet_folder, final_layer_size, num_tasks, test_subset)
        if not multitask:
            num_classes = 2
        else:
            if 'same_visit_8192' in label_column:
                num_classes = 8192
            else:
                num_classes = len(label_column)
        model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=num_classes).to(device)
        
        if loadmodel_path:
            state_dict = torch.load(loadmodel_path, map_location=f'cuda:{torch.cuda.current_device()}')
            state_dict = convert_3d_to_2d_weights_densenet(state_dict, model) 
            model = load_different_model(model, state_dict, ddp)
            
            path_ls = loadmodel_path.split('_')
            for chunk in path_ls:
                if 'epoch' in chunk:
                    starting_epoch = int(chunk.replace('epoch', ''))  
                    break
                
        # Load the pretrained weights
        elif (not loadmodel_path) and pretrained_path_densenet:
            
            state_dict = torch.load(pretrained_path_densenet, map_location=f'cuda:{torch.cuda.current_device()}') 
            state_dict = convert_3d_to_2d_weights_densenet(state_dict, model)
            model = load_different_model(model, state_dict, ddp)
            
            starting_epoch = -1
            
        if linear_probe:
            print("Linear probe for densenet...")
            # Freeze all layers in the model
            for name, param in model.named_parameters():
                if "features" in name:
                    param.requires_grad = False
                    
        if pretrained_path_densenet:
            model_choice += "_600k"
        
        if ddp:
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
            
        if dataformat == "nii_gz_RSNAPE":
            model_choice += "_RSNAPE"  
            
        if use_crop:
            model_choice += "_crop"
            
            
    elif model_choice == "resnet_louis":
        motor_task, tokenizer, train_dataset, valid_dataset, test_dataset, processor, index_train, index_valid, index_test, num_tasks, ontology = set_up_motor_task(TARGET_DIR, from_pretrained_tokenizer, month_date_hour, num_proc, label_csv, ontology_path, inference, vocab_size, START_TIME, parquet_folder, final_layer_size, num_tasks, test_subset)
        if not multitask:
            num_classes = 2
        else:
            if 'same_visit_8192' in label_column:
                num_classes = 8192
            else:
                num_classes = len(label_column)
        print(f"Building model imagenet_stage1_stage2_mtl_seg_resnet_100 with {num_classes} classes")
        
        model = resnet152(n_input_channels=1)
        checkpoint = torch.load(pretrained_path_resnet, map_location=f'cuda:{torch.cuda.current_device()}')
        checkpoint_updated = {}
        for key, value in checkpoint.items():
            if "encode_image.i3_resnet." in key:
                key = key.replace("encode_image.i3_resnet.", "")
                checkpoint_updated[key] = value
            else:
                checkpoint_updated[key] = value
        model_state_dict = model.state_dict()
        filtered_checkpoint = {k: v for k, v in checkpoint_updated.items() if k in model_state_dict and model_state_dict[k].size() == v.size()}
        model.load_state_dict(filtered_checkpoint, strict=False)

        starting_epoch = -1            
        if ddp:
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False) 
            
        if dataformat == "nii_gz_RSNAPE":
            model_choice += "_RSNAPE"  
            
        if use_crop:
            model_choice += "_crop"
            
    elif model_choice == "unet_louis":
        pretrained_path = pretrained_path_swinUNETR
        model = SwinClassifier(
        spatial_dims=3, in_channels=1, out_channels=2, pretrained_path=pretrained_path
    ).to(device)
        if loadmodel_path:
            state_dict = torch.load(loadmodel_path)
            # model.load_state_dict(state_dict)
            model = load_different_model(model, state_dict, ddp)
        
        if linear_probe:
            print("Linear probe for SwinUNETR...")
            # Freeze all layers in the model
            for name, param in model.named_parameters():
                if 'swin_unetr' in name and "out" not in name:
                    param.requires_grad = False
                    
        if ddp:
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)       
    
    elif model_choice == "unet_louis_tte":
        motor_task, tokenizer, train_dataset, valid_dataset, test_dataset, processor, index_train, index_valid, index_test, num_tasks, ontology = set_up_motor_task(TARGET_DIR, from_pretrained_tokenizer, month_date_hour, num_proc, label_csv, ontology_path, inference, vocab_size, START_TIME, parquet_folder, final_layer_size, num_tasks, test_subset)
        if only_train_tokenizer:
            exit()
            
        print('number of tasks: ', len(motor_task.pretraining_task_info))
            
        pretrained_path = pretrained_path_swinUNETR
        model = SwinClassifier_TTE(time_bins=motor_task.time_bins, pretraining_task_info=motor_task.get_task_config().task_kwargs['pretraining_task_info'], device=device, final_layer_size=motor_task.final_layer_size, use_checkpoint=use_checkpoint,
        spatial_dims=3, in_channels=1, out_channels=2, pretrained_path=pretrained_path
        ).to(device)
        if loadmodel_path:
            state_dict = torch.load(loadmodel_path)
            # model.load_state_dict(state_dict)
            model = load_different_model(model, state_dict, ddp)
        
        if linear_probe:
            print("Linear probe for SwinUNETR...")
            # Freeze all layers in the model
            for name, param in model.named_parameters():
                if 'swin_unetr' in name and "out" not in name:
                    param.requires_grad = False
     
        if ddp:
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
    elif model_choice == "swin":
        model = SwinClassifier(spatial_dims=3, in_channels=1, out_channels=2).to(device)
        if loadmodel_path:
            state_dict = torch.load(loadmodel_path)
            model.load_state_dict(state_dict)
    
    elif model_choice == 'resnetv2_mars':
        motor_task, tokenizer, train_dataset, valid_dataset, test_dataset, processor, index_train, index_valid, index_test, num_tasks, ontology = set_up_motor_task(TARGET_DIR, from_pretrained_tokenizer, month_date_hour, num_proc, label_csv, ontology_path, inference, vocab_size, START_TIME, parquet_folder, final_layer_size, num_tasks, test_subset)
        if not multitask:
            num_classes = 2
        else:
            if 'same_visit_8192' in label_column:
                num_classes = 8192
            else:
                num_classes = len(label_column)
        
        model = ResNetV2_Mars().to(device)
        checkpoint = torch.load(loadmodel_path, map_location=f'cuda:{torch.cuda.current_device()}') 
        ckpt = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
        # model.load_state_dict(ckpt, strict=False)
        model = load_different_model(model, ckpt, ddp)

        starting_epoch = -1     
        if ddp:
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
            
        if dataformat == "nii_gz_RSNAPE":
            model_choice += "_RSNAPE"  
            
        if use_crop:
            model_choice += "_crop"
            
    else:
        raise ValueError("Model choice not recognized.", model_choice)

    # CrossEntropyLoss and Adam optimizer
    loss_function = torch.nn.CrossEntropyLoss()
    # loss_function = torch.nn.BCEWithLogitsLoss()  # also works with this data

    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1)
    scaler = GradScaler() 
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    # start a typical PyTorch training
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    writer = SummaryWriter()
    ##################################################################################################################

    # pretraining TTE
    if not inference and ('tte' in model_choice):
        model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
        
        for epoch in range(starting_epoch + 1, starting_epoch + max_epochs):
            print("-" * 10)
            print(f"Current epoch {epoch} to train to epoch {starting_epoch + max_epochs -1}")
            model.train()
            epoch_loss = 0
            epoch_loss_valid = 0
            step = -1
            X_train = []
            image_paths_train = []
            y_train = []
            no_followup = 0
            no_followup_valid = 0
            no_crop_train = 0
            
            train_iter = iter(train_loader)
            
            # while True:
            for batch_data_idx in tqdm(range(len(train_loader)), total=len(train_loader), ncols=75):
                step += 1
                try:
                    batch_data = next(train_iter)
                except FileNotFoundError:
                    no_crop_train += 1
                    continue
                
                if use_cachedataset:
                    if 'image_path' in batch_data:
                        inputs, labels, image_path = batch_data['image'].to(device), batch_data['label'], batch_data['image_path']
                    else:
                        inputs, labels,  = batch_data['image'].to(device), batch_data['label'], 
                        image_path = image_paths_train_[step]
                else:
                    inputs, labels = batch_data[0].to(device), batch_data[1].to(device)

                patient_id = int(image_path[0].replace(nii_folder+ "/", '').split('_')[0])
                ct_time = ' '.join(image_path[0].replace(nii_folder+ "/", '').replace('.nii.gz', '').split('_')[1:])
                ct_time = datetime.strptime(ct_time, '%Y-%m-%d %H %M %S')
                
                # if step % 100 == 0:
                #     print(f"Current step: {step}, time used: {datetime.now() - START_TIME}")
                
                
                final_batch, _ = get_final_batch(image_path, nii_folder, motor_task, train_dataset, index_train, ontology)
                
                # print(final_batch['is_event'].shape)
                        
                if final_batch['is_event'].shape[0] == 0:
                    no_followup += 1
                    continue

                optimizer.zero_grad()
                
                if mixed_precision:
                    # Enable autocast for the forward pass
                    with autocast():
                        loss, _, time_independent_features = model(inputs, final_batch, return_logits=False)
                    # Backward pass and optimization are handled with GradScaler
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                
                    loss, _, time_independent_features = model(inputs, final_batch, return_logits=False)
                    # loss.backward()
                    accelerator.backward(loss)
                    optimizer.step()
                
                
                # print('no followup:', no_followup, end='\r')  
                epoch_loss += loss.item()
                epoch_len = len(train_ds) // batch_size                
                # print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}, time_used: {time_used}", end="  ")
                # writer.add_scalar("train_loss", loss.item(), epoch_len * (epoch-1) + step)
                # run.track(loss.item(), name='train_loss', step=epoch_len * (epoch-1) + step, context={'subset': 'train'})
                
                # collect train set features
                # features = model(inputs, example_batch['batch'],  inference=True)
                # features = features.cpu().detach().numpy()
                features = time_independent_features.cpu().detach().numpy()
                # features = np.squeeze(features)
                # labels = labels.cpu().detach().numpy()
                # labels = np.squeeze(labels)
                # labels = labels[:, 0]
                X_train.append(features)
                image_paths_train.append(image_path)
                y_train.append(labels)
                # except EOFError:
                #     print(f"Error in patient {patient_id} at time {ct_time}, no cachedataset")
                #     continue
                # except StopIteration:
                #     print(f"StopIteration at patient {patient_id} at time {ct_time}")
                #     break
                # except Exception as e:
                #     print(f"Error in patient {patient_id} at time {ct_time}, {e}")
                #     continue
            
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']  
            # run.track(current_lr, name='current_lr', step=epoch, context={'subset': 'train'})  
            if (ddp and rank == 0) or (not ddp):    
                run_wandb.log({'current_lr': current_lr})
            scheduler.step(epoch_loss)    
            # print('no followup:', no_followup)      

            writer.add_scalar("epoch_loss", epoch_loss, epoch)
            # run.track(epoch_loss, name='epoch_loss', step=epoch, context={'subset': 'train'})
            if (ddp and rank == 0) or (not ddp):
                run_wandb.log({'epoch_loss': epoch_loss})
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch} average loss: {epoch_loss:.4f}")
            # run.track(epoch_loss, name='epoch_loss_average', step=epoch, context={'subset': 'train'})
            if (ddp and rank == 0) or (not ddp):
                run_wandb.log({'epoch_loss_average': epoch_loss})
            
            month_date = datetime.now().strftime("%m%d")
            if (not prop_train) and (not prop_valid):
                print('model saving at:', os.path.join(model_save_path, f"best_metric_model_{epoch}epoch_{model_choice}_{month_date}.pth"))
                torch.save(
                    model.state_dict(), os.path.join(model_save_path, f"best_metric_model_{epoch}epoch_{model_choice}_{month_date}.pth")
                )
            
            if (epoch + 1) % val_interval == 0:
                model.eval()

                num_correct = 0.0
                metric_count = 0
                proba = []
                labels = []
                X_val = []
                image_paths_val = []
                y_val = []
                step_valid = -1
                
                for val_data in tqdm(val_loader, ncols=75):
                    step_valid += 1
                    
                    # patient_id = int(image_paths_valid[step].replace(nii_folder+ "/", '').split('_')[0])
                    # ct_time = ' '.join(image_paths_valid[step].replace(nii_folder+ "/", '').replace('.nii.gz', '').split('_')[1:])
                    # ct_time = datetime.strptime(ct_time, '%Y-%m-%d %H %M %S')
                    # for idx, event in enumerate(valid_dataset[index_valid.get_index(patient_id)]['events']):
                    #     if event['time'] == ct_time:
                    #         offset = idx
                    
                    # example_batch = processor.collate([processor.convert_patient(train_dataset[index_valid.get_index(patient_id)], tensor_type='pt', offset=offset, max_patient_length=vocab_size)])
                    
                    
                    if not use_cachedataset:
                        val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    else:
                        if 'image_path' in val_data:
                            val_images, val_labels, image_path = val_data['image'].to(device), val_data['label'], val_data['image_path']
                        else:
                            val_images, val_labels = val_data['image'].to(device), val_data['label']
                            image_path = image_paths_valid[step_valid]
                    with torch.no_grad():
                        final_batch, _ = get_final_batch(image_path, nii_folder, motor_task, valid_dataset, index_valid, ontology)
                        
                        if final_batch['is_event'].shape[0] == 0:
                            no_followup_valid += 1
                            continue
                        
                        loss, _, time_independent_features = model(val_images, final_batch)
                        X_val.append(time_independent_features.cpu().detach().numpy())
                        image_paths_val.append(image_path)
                        y_val.append(val_labels)
                        
                       
                    epoch_loss_valid += loss.item()    
                    # print(loss.item(), end=' ')
                writer.add_scalar("epoch_loss_valid", epoch_loss_valid, epoch)
                # run.track(epoch_loss_valid, name='epoch_loss_valid', step=epoch, context={'subset': 'valid'})
                if (ddp and rank == 0) or (not ddp):
                    run_wandb.log({'epoch_loss_valid': epoch_loss_valid})
                epoch_loss_valid /= step_valid
                print(f"epoch {epoch} average valid loss: {epoch_loss_valid:.4f}")
                # run.track(epoch_loss_valid, name='epoch_loss_valid_average', step=epoch, context={'subset': 'valid'})
                if (ddp and rank == 0) or (not ddp):
                    run_wandb.log({'epoch_loss_valid_average': epoch_loss_valid})
                print('no followup valid:', no_followup_valid)
                
                
                X_val = np.concatenate(X_val, axis=0)
                image_paths_val = np.concatenate(image_paths_val, axis=0)
                y_val = np.concatenate(y_val, axis=1).transpose()
                X_train = np.concatenate(X_train, axis=0)
                image_paths_train = np.concatenate(image_paths_train, axis=0)
                y_train = np.concatenate(y_train, axis=1).transpose()
                
                metric_values, _, _, auroc_val, auroc_train_dict, auroc_val_dict = logistic_regression(label_column, X_train, y_train, X_val, y_val, model_save_path, model_choice, epoch, month_date, metric_values, None, None, tune_linearprobe)
                for key, value in auroc_train_dict.items():
                    # run.track(value, name=key, step=epoch, context={'subset': 'train'})
                    if (ddp and rank == 0) or (not ddp):
                        run_wandb.log({key: value})
                    writer.add_scalar(key, value, epoch)
                for key, value in auroc_val_dict.items():
                    # run.track(value, name=key, step=epoch, context={'subset': 'valid'})
                    if (ddp and rank == 0) or (not ddp):
                        run_wandb.log({key: value})
                    writer.add_scalar(key, value, epoch)
                
                # save model
                if auroc_val > best_metric and (not prop_train) and (not prop_valid):
                    best_metric = auroc_val
                    best_metric_epoch = epoch
                    print("model saving at:" + os.path.join(model_save_path, f"best_metric_model_{epoch}epoch_{model_choice}.pth"))
                    torch.save(
                        model.state_dict(), os.path.join(model_save_path, f"best_metric_model_{epoch}epoch_{model_choice}.pth")
                    )
                    

                print(f"Current epoch: {epoch+1} current accuracy: {auroc_val:.4f} ")
                print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
                
                
                # save train features
                if not prop_train:
                    with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"X_train_{epoch}epoch_{model_choice}.pkl"), 'wb') as f:
                        pickle.dump(X_train, f)
                    with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"image_paths_train_{epoch}epoch_{model_choice}.pkl"), 'wb') as f:
                        pickle.dump(image_paths_train, f)
                    with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"y_train_{epoch}epoch_{model_choice}.pkl"), 'wb') as f:
                        pickle.dump(y_train, f)

                # save val features
                if not prop_valid:
                    with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"X_val_{epoch}epoch_{model_choice}.pkl"), 'wb') as f:
                        pickle.dump(X_val, f)
                    with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"image_paths_val_{epoch}epoch_{model_choice}.pkl"), 'wb') as f:
                        pickle.dump(image_paths_val, f)
                    with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"y_val_{epoch}epoch_{model_choice}.pkl"), 'wb') as f:
                        pickle.dump(y_val, f)
                    
        print(
        f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
        )
        writer.close()

        time_used = datetime.now() - START_TIME
        print(time_used)

    
    # multitask pretraining 
    elif not inference and ('_tte' not in model_choice) and multitask:
        loss_function = torch.nn.BCEWithLogitsLoss() 
        # loss_function = torch.nn.BCELoss()
        
        model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
        
        for epoch in range(starting_epoch + 1, starting_epoch + max_epochs):
            print("-" * 10)
            print(f"Current epoch {epoch} to train to epoch {starting_epoch + max_epochs -1}")
            model.train()
            epoch_loss = 0
            epoch_loss_valid = 0
            step = -1
            X_train = []
            image_paths_train = []
            y_train = []
            y_proba_train = []
            no_followup = 0
            no_followup_valid = 0
            
            # while True:
            for batch_data in tqdm(train_loader, ncols=75):
                step += 1
                if use_cachedataset:
                    if 'image_path' in batch_data:
                        inputs, labels, image_path = batch_data['image'].to(device), batch_data['label'], batch_data['image_path']
                    else:
                        inputs, labels,  = batch_data['image'].to(device), batch_data['label'], 
                        image_path = image_paths_train_[step]
                        
                    if not labels:
                        continue
                    
                    if 'same_visit_8192' not in label_column:
                        try:
                            labels = [-1 if label[0] == 'Censored' else int(label[0]) for label in labels]
                        except:
                            labels = convert_to_list(labels)
                            labels = [-1 if label == 'Censored' else int(label) for label in labels]
                    
                    try:
                        if labels[0].is_cuda:
                            labels_ = []
                            for label in labels:
                                labels_.append(label.cpu().detach().numpy())
                            labels = labels_
                    except:
                        pass
                    
                    labels = torch.from_numpy(np.array(labels, dtype=int)).to(device)
                    labels = labels.unsqueeze(1)
                   
                else:
                    inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                

                optimizer.zero_grad()
                
                # if mixed_precision:
                if False:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    # Enable autocast for the forward pass
                    with autocast():
                        # gradient overflow issue
                        def initialize_weights(m):
                            if isinstance(m, nn.Linear):
                                torch.nn.init.xavier_normal_(m.weight)
                        model.apply(initialize_weights)
                        
                        outputs, train_features = model(inputs, return_features=True, multitask=multitask)
                    # Backward pass and optimization are handled with GradScaler
                    if 'same_visit_8192' in label_column:
                        if type(labels) == list:
                            labels = torch.tensor(labels, device=outputs.device, dtype=outputs.dtype)
                            print(type(labels), labels)
                        try:
                            # ddp
                            loss = loss_function(outputs.squeeze(), labels.squeeze().float())
                        except:
                            # single gpu
                            loss = loss_function(torch.tensor(outputs, device=device), torch.tensor(labels, device=device).seqeeze().float())
                    else:
                        loss = loss_function(outputs[0].squeeze(), labels[0].squeeze().float())
                        for i in range(1, labels.shape[1]):
                            if labels[i].item() == -1:
                                mask = torch.tensor([0.0], device=loss.device, dtype=loss.dtype)
                            else:
                                mask = torch.tensor([1.0], device=loss.device, dtype=loss.dtype)
                            loss += loss_function(outputs[i].squeeze(), labels[i].squeeze()) * mask
                    print(loss, 'loss -----------------------')
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                
                # no mixed precision (generally not used)
                if True:
                    outputs, train_features = model(inputs, return_features=True, multitask=multitask)
                    
                    if 'same_visit_8192' in label_column:
                        if type(labels) == list:
                            labels = torch.tensor(labels, device=outputs.device, dtype=outputs.dtype)
                            print(type(labels), labels)
                        try:
                            loss = loss_function(outputs.squeeze(), labels.squeeze().float())
                        except Exception as e:
                            print(e)
                            print(outputs[:10], labels[:10])
                            print(type(outputs), type(labels))
                            print(outputs[0].shape, labels[0].shape)
                            # outputs_ = [o.cpu().detach().numpy() for o in outputs]
                            # labels_ = [l.cpu().detach().numpy() for l in labels]
                            # loss = loss_function(torch.tensor(np.array(outputs_).squeeze(), device=device), torch.tensor(np.array(labels_), device=device).squeeze().float())
                    
                    # normal MTL, not same_visit_8192
                    else:
                        # loss = loss_function(outputs[0].squeeze(), labels[0].squeeze().float())
                        loss = torch.tensor(0.0, device=outputs[0].device, dtype=outputs[0].dtype)
                        for i in range(0, labels.shape[1]):
                            if labels[i].item() == -1:
                                mask = torch.tensor([0.0], device=outputs[0].device, dtype=outputs[0].dtype)
                            else:
                                mask = torch.tensor([1.0], device=outputs[0].device, dtype=outputs[0].dtype)
                            loss += loss_function(outputs[i].squeeze(), labels[i].squeeze().float()) * mask.squeeze()
                            
                    # loss.backward()
                    accelerator.backward(loss)
                    # optimizer.step()
                    if step + 1 % accumulation_steps == 0:
                        optimizer.step()  # Perform optimization step after accumulating gradients
                        optimizer.zero_grad()  # Reset gradients after each step    
                
                epoch_loss += loss.item()
                epoch_len = len(train_ds) // batch_size                
                
                # collect train set features

                features = train_features.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()

                X_train.append(features)
                image_paths_train.append(image_path)
                y_train.append(labels)
                y_proba_train.append([output.cpu().detach().numpy() for output in outputs])

            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']  
            # run.track(current_lr, name='current_lr', step=epoch, context={'subset': 'train'})    
            if (ddp and rank == 0) or (not ddp):
                run_wandb.log({'current_lr': current_lr})
            scheduler.step(epoch_loss)    
            # print('no followup:', no_followup)      

            writer.add_scalar("epoch_loss", epoch_loss, epoch)
            # run.track(epoch_loss, name='epoch_loss', step=epoch, context={'subset': 'train'})
            if (ddp and rank == 0) or (not ddp):
                run_wandb.log({'epoch_loss': epoch_loss})
            step += 1 # bc starting from -1
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch} average loss: {epoch_loss:.4f}")
            # run.track(epoch_loss, name='epoch_loss_average', step=epoch, context={'subset': 'train'})
            if (ddp and rank == 0) or (not ddp):
                run_wandb.log({'epoch_loss_average': epoch_loss})
            month_date = datetime.now().strftime("%m%d")
            if step > 10 and (not prop_train) and (not prop_valid):
                print('model saving at:', os.path.join(model_save_path, f"best_metric_model_{epoch}epoch_{model_choice}_{month_date}.pth"))
                torch.save(
                    model.state_dict(), os.path.join(model_save_path, f"best_metric_model_{epoch}epoch_{model_choice}_{month_date}.pth")
                )
            
            if (epoch + 1) % val_interval == 0:
                model.eval()

                num_correct = 0.0
                metric_count = 0
                proba = []
                labels = []
                X_val = []
                image_paths_val = []
                y_val = []
                y_proba_val = []
                step_valid = -1
                
                for val_data in tqdm(val_loader, ncols=75):
                    step_valid += 1

                    if not use_cachedataset:
                        val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    else:
                        if 'image_path' in val_data:
                            val_images, val_labels, image_path = val_data['image'].to(device), val_data['label'], val_data['image_path']
                        else:
                            val_images, val_labels = val_data['image'].to(device), val_data['label']
                            image_path = image_paths_valid[step_valid]
                        if not val_labels:
                            continue    
                        if 'same_visit_8192' not in label_column:
                            try:
                                val_labels = [-1 if label[0] == 'Censored' else int(label[0]) for label in val_labels]
                            except:
                                val_labels = convert_to_list(val_labels)
                                val_labels = [-1 if label == 'Censored' else int(label) for label in val_labels]
                        
                        try:
                            if val_labels[0].is_cuda:
                                val_labels_ = []
                                for label in val_labels:
                                    val_labels_.append(label.cpu().detach().numpy())
                                val_labels = val_labels_
                        except:
                            pass

                        val_labels = torch.from_numpy(np.array(val_labels, dtype=int)).to(device)
                        val_labels = val_labels.unsqueeze(1)
                    with torch.no_grad():
                        outputs, val_features = model(val_images, return_features=True, multitask=multitask)

                        if 'same_visit_8192' in label_column:
                            if type(val_labels) == list:
                                val_labels = torch.tensor(val_labels, device=outputs.device, dtype=outputs.dtype)
                                print(type(val_labels), val_labels)
                            try:
                                loss = loss_function(outputs.squeeze(), val_labels.squeeze().float())
                            except:
                                outputs_ = [o.cpu() for o in outputs]
                                loss = loss_function(torch.tensor(np.array(outputs_), device=device), torch.tensor(np.array(val_labels), device=device).seqeeze().float())
                        else:
                            # loss = loss_function(outputs[0].squeeze(), val_labels[0].squeeze().float())
                            loss = torch.tensor(0.0, device=outputs[0].device, dtype=outputs[0].dtype)
                            for i in range(0, val_labels.shape[1]):
                                if val_labels[i].item() == -1:
                                    mask = torch.tensor([0.0], device=outputs[0].device, dtype=outputs[0].dtype)
                                else:
                                    mask = torch.tensor([1.0], device=outputs[0].device, dtype=outputs[0].dtype)
                                loss += loss_function(outputs[i].squeeze(), val_labels[i].squeeze().float()) * mask.squeeze()
                
                        X_val.append(val_features.cpu().detach().numpy())
                        image_paths_val.append(image_path)
                        y_val.append(val_labels.cpu().detach().numpy())
                        y_proba_val.append([output.cpu().detach().numpy() for output in outputs])
                        
                    epoch_loss_valid += loss.item()    
                    # print(loss.item(), end=' ')
                writer.add_scalar("epoch_loss_valid", epoch_loss_valid, epoch)
                # run.track(epoch_loss_valid, name='epoch_loss_valid', step=epoch, context={'subset': 'valid'})
                if (ddp and rank == 0) or (not ddp):
                    run_wandb.log({'epoch_loss_valid': epoch_loss_valid})
                epoch_loss_valid /= step_valid
                print(f"epoch {epoch} average valid loss: {epoch_loss_valid:.4f}")
                # run.track(epoch_loss_valid, name='epoch_loss_valid_average', step=epoch, context={'subset': 'valid'})
                if (ddp and rank == 0) or (not ddp):
                    run_wandb.log({'epoch_loss_valid_average': epoch_loss_valid})
                print('no followup valid:', no_followup_valid)
                
                X_val = np.concatenate(X_val, axis=0)
                try:
                    image_paths_val = np.concatenate(image_paths_val, axis=0)
                except:
                    image_paths_val = np.concatenate(np.array(image_paths_val).reshape(-1,1), axis=0)
                y_val = np.concatenate(y_val, axis=1).transpose()
                X_train = np.concatenate(X_train, axis=0)
                try:
                    image_paths_train = np.concatenate(image_paths_train, axis=0)
                except:
                    image_paths_train = np.concatenate(np.array(image_paths_train).reshape(-1,1), axis=0)
                y_train = np.concatenate(y_train, axis=1).transpose()
                
                metric_values, _, _, auroc_val, auroc_train_dict, auroc_val_dict = logistic_regression(label_column, X_train, y_train, X_val, y_val, model_save_path, model_choice, epoch, month_date, metric_values, None, None, tune_linearprobe)
                for key, value in auroc_train_dict.items():
                    # run.track(value, name=key, step=epoch, context={'subset': 'train'})
                    if (ddp and rank == 0) or (not ddp):
                        run_wandb.log({key: value})
                    # writer.add_scalar(key, value, epoch)
                for key, value in auroc_val_dict.items():
                    # run.track(value, name=key, step=epoch, context={'subset': 'valid'})
                    if (ddp and rank == 0) or (not ddp):
                        run_wandb.log({key: value})
                    # writer.add_scalar(key, value, epoch)
                
                # save model
                if auroc_val > best_metric and (not prop_train) and (not prop_valid):
                    best_metric = auroc_val
                    best_metric_epoch = epoch
                    print("model saving at:" + os.path.join(model_save_path, f"best_metric_model_{epoch}epoch_{model_choice}.pth"))
                    torch.save(
                        model.state_dict(), os.path.join(model_save_path, f"best_metric_model_{epoch}epoch_{model_choice}.pth")
                    )
                    

                print(f"Current epoch: {epoch+1} current accuracy: {auroc_val:.4f} ")
                print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")

                # AUROC
                y_proba_train = np.concatenate(y_proba_train, axis=1).transpose().squeeze().reshape(-1, len(label_column))
                y_proba_val = np.concatenate(y_proba_val, axis=1).transpose().squeeze().reshape(-1, len(label_column))
                for idx, task in enumerate(label_column):
                    if task == 'same_visit_8192':
                        continue
                    try:
                        idx_uncensored = np.where((y_train != 'Censored') & (y_train != -1))[0]
                        auroc_train = sklearn.metrics.roc_auc_score(y_train[idx_uncensored, idx], y_proba_train[idx_uncensored, idx])

                        idx_uncensored_val = np.where((y_val != 'Censored') & (y_val != -1))[0]
                        auroc_val = sklearn.metrics.roc_auc_score(y_val[idx_uncensored_val, idx], y_proba_val[idx_uncensored_val, idx])
                        print(f"=====NN model output: Train {task} AUROC: {auroc_train:.4f}, Val {task} AUROC: {auroc_val:.4f}")
                    except:
                        print(f"Error in {task}")
                        continue
                
                
                # save train features
                if not prop_train:
                    # check folder exists
                    if not os.path.exists(model_save_path.replace('model_checkpoints', 'features')):
                        os.makedirs(model_save_path.replace('model_checkpoints', 'features'))
                    
                    with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"X_train_{epoch}epoch_{model_choice}.pkl"), 'wb') as f:
                        pickle.dump(X_train, f)
                    with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"image_paths_train_{epoch}epoch_{model_choice}.pkl"), 'wb') as f:
                        pickle.dump(image_paths_train, f)
                    with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"y_train_{epoch}epoch_{model_choice}.pkl"), 'wb') as f:
                        pickle.dump(y_train, f)
                    with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"y_proba_train_{epoch}epoch_{model_choice}.pkl"), 'wb') as f:
                        pickle.dump(y_proba_train, f)

                # save val features
                if not prop_valid:
                    with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"X_val_{epoch}epoch_{model_choice}.pkl"), 'wb') as f:
                        pickle.dump(X_val, f)
                    with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"image_paths_val_{epoch}epoch_{model_choice}.pkl"), 'wb') as f:
                        pickle.dump(image_paths_val, f)
                    with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"y_val_{epoch}epoch_{model_choice}.pkl"), 'wb') as f:
                        pickle.dump(y_val, f)
                    with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"y_proba_val_{epoch}epoch_{model_choice}.pkl"), 'wb') as f:
                        pickle.dump(y_proba_val, f)

            early_stopping(epoch_loss_valid)
            if early_stopping.early_stop:
                print("Early stopping triggered. Stopping training.")
                break

        print(
        f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
        )
        writer.close()

        time_used = datetime.now() - START_TIME
        print(time_used)
    

    # direct supervision, no TTE pretraining
    elif inference == False and (not "_tte" in model_choice):
        model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
        for epoch in range(starting_epoch + 1, starting_epoch + max_epochs):
            print("-" * 10)
            print(f"Current epoch {epoch} to train to epoch {starting_epoch + max_epochs -1}")
            model.train()
            epoch_loss = 0
            step = -1
            train_iter = iter(train_loader)
            censor_count = 0
            X_train = []
            image_paths_train = []
            y_train = []
            print('len of train_loader:', len(train_loader))
            while True:
                try:
                    batch_data = next(train_iter)
                    step += 1
                    if step % 100 == 0:
                        print(f"Current step: {step}, time used: {datetime.now() - START_TIME}")
                    if use_cachedataset:
                        if 'image_path' in batch_data:
                            # inputs, labels, image_path = batch_data['image'].to(device), batch_data['finetune_label'], batch_data['image_path']
                            inputs, labels, image_path = batch_data['image'].to(device), batch_data['label'], batch_data['image_path']
                        else:
                            # inputs, labels,  = batch_data['image'].to(device), batch_data['finetune_label'], 
                            inputs, labels,  = batch_data['image'].to(device), batch_data['label'], 
                            image_path = image_paths_train_[step]
                        try:
                            labels = torch.from_numpy(np.array(labels, dtype=int)).to(device)
                        except (TypeError, ValueError) as e:
                            censor_count += 1
                            continue
                    else:
                        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                    optimizer.zero_grad()
                    outputs, train_features = model(inputs, return_features=True)
                    # print('train_features:', train_features.shape)
                    # print('labels:', labels.shape)
                    if train_features.shape[0] == batch_size and labels.shape[0] == batch_size:
                        for i in range(train_features.shape[0]):
                            X_train.append([train_features[i].cpu().detach().numpy()])
                            image_paths_train.append(image_path[i])
                            y_train.append(labels[i].cpu().detach().numpy())
                    else:
                        X_train.append(train_features.cpu().detach().numpy())
                        image_paths_train.append(image_path)
                        y_train.append(labels.cpu().detach().numpy())
                    # if labels == 'Censored':
                    #     continue
                    # labels = torch.tensor(np.array(labels, dtype=int), dtype=torch.float32).to(device)
                    loss = loss_function(outputs.squeeze(), labels.squeeze())
                    # loss.backward()
                    accelerator.backward(loss)
                    optimizer.step()
                    epoch_loss += loss.item()
                    epoch_len = len(train_ds) // batch_size
                    time_used = datetime.now() - START_TIME
                    # print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}, time_used: {time_used}", end="  ")
                    writer.add_scalar("train_loss", loss.item(), epoch_len * (epoch-1) + step)
                    # run.track(loss.item(), name='train_loss', step=epoch_len * (epoch-1) + step, context={'subset': 'train'})
                except StopIteration:
                    print(f"StopIteration at step {step}")
                    break
                except EOFError:
                    print(f"Error in step {step}, {image_path}, no cachedataset")
                    continue
                # except Exception as e:
                #     print(f"Error in step {step}, {e}")
                #     break
            
            
            X_train = np.concatenate(X_train, axis=0)
            image_paths_train = np.concatenate(np.array(image_paths_train).reshape(-1, 1), axis=0)
            y_train = np.array(y_train)
            # if len(y_train.shape) > 1:
            #         y_train = np.concatenate(y_train, axis=1).transpose()
            # elif len(y_train.shape) == 1:
            y_train = np.expand_dims(y_train, axis=0)
            y_train = y_train.transpose()
            print('X_train after:', X_train.shape)
            print('y_train after:', y_train.shape, np.unique(y_train, return_counts=True))
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']  
            # run.track(current_lr, name='current_lr', step=epoch, context={'subset': 'train'})   
            if (ddp and rank == 0) or (not ddp):
                run_wandb.log({'current_lr': current_lr}) 
            scheduler.step(epoch_loss)    
            writer.add_scalar("epoch_loss", epoch_loss, epoch)    
            # run.track(epoch_loss, name='epoch_loss', step=epoch, context={'subset': 'train'})
            if (ddp and rank == 0) or (not ddp):
                run_wandb.log({'epoch_loss': epoch_loss})
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)   
            print('censor_count direct supervision:', censor_count, 'out of', step+1)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
            if (epoch + 1) % val_interval == 0:
                model.eval()
                censor_count_valid = 0
                num_correct = 0.0
                metric_count = 0
                proba = []
                labels = []
                X_val = []
                image_paths_val = []
                y_val = []
                epoch_loss_valid = 0
                print('len of val_loader:', len(val_loader))
                for val_data in tqdm(val_loader, ncols=75):
                    if not use_cachedataset:
                        val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    else:
                        # val_images, val_labels = val_data['image'].to(device), val_data['finetune_label']
                        val_images, val_labels = val_data['image'].to(device), val_data['label']
                        try:
                            val_labels = torch.from_numpy(np.array(val_labels, dtype=int)).to(device)
                        except (TypeError, ValueError) as e:
                            censor_count_valid += 1
                            continue
                    with torch.no_grad():
                        val_outputs, val_features = model(val_images, return_features=True)
                        
                        if val_features.shape[0] == batch_size and val_labels.shape[0] == batch_size:
                            for i in range(val_features.shape[0]):
                                X_val.append([val_features[i].cpu().detach().numpy()])
                                image_paths_val.append(val_data['image_path'][i])
                                y_val.append(val_labels[i].cpu().detach().numpy())
                        else:
                            X_val.append(val_features.cpu().detach().numpy())
                            image_paths_val.append(val_data['image_path'])
                            y_val.append(val_labels.cpu().detach().numpy())

                        loss_valid = loss_function(val_outputs.squeeze(), val_labels.squeeze())
                        epoch_loss_valid += loss_valid.item()
                        value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                        metric_count += len(value)
                        num_correct += value.sum().item()
                        proba.extend(val_outputs[:,1].cpu())
                        labels.extend(val_labels.cpu())
                        
                labels = np.squeeze(labels)

                X_val = np.concatenate(X_val, axis=0)
                image_paths_val = np.concatenate(np.array(image_paths_val).reshape(-1, 1), axis=0)
                y_val = np.array(y_val)
                print('X_val after:', X_val.shape)
                print('y_val after:', y_val.shape, np.unique(y_val, return_counts=True))
                # if len(y_val.shape) > 1:
                #     y_val = np.concatenate(y_val, axis=1).transpose()
                # elif len(y_val.shape) == 1:
                y_val = np.expand_dims(y_val, axis=0)
                y_val = y_val.transpose()
                month_date = datetime.now().strftime("%m%d")
                metric_values, _, _, auroc_val, auroc_train_dict, auroc_val_dict = logistic_regression([finetune_label], X_train, y_train, X_val, y_val, model_save_path, model_choice, epoch, month_date, metric_values, None, None, tune_linearprobe)
                for key, value in auroc_train_dict.items():
                    # run.track(value, name=key, step=epoch, context={'subset': 'train'})
                    if (ddp and rank == 0) or (not ddp):
                        run_wandb.log({key: value})
                    writer.add_scalar(key, value, epoch)
                for key, value in auroc_val_dict.items():
                    # run.track(value, name=key, step=epoch, context={'subset': 'valid'})
                    if (ddp and rank == 0) or (not ddp):
                        run_wandb.log({key: value})
                    writer.add_scalar(key, value, epoch)
                print('censor_count valid:', censor_count_valid, 'out of', len(val_loader))        
                # metric = num_correct / metric_count
                metric = roc_auc_score(labels, proba)
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    # save model no matter what
                    print('model saving at:',  os.path.join(model_save_path, f"best_metric_model_{epoch}epoch_{model_choice}_{finetune_label}{'_lb' if linear_probe else ''}.pth"))
                    torch.save(
                    model.state_dict(), os.path.join(model_save_path, f"best_metric_model_{epoch}epoch_{model_choice}_{finetune_label}{'_lb' if linear_probe else ''}.pth")
                    )
                print(f"Current epoch: {epoch+1} current accuracy: {metric:.4f} ")
                print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
                writer.add_scalar("val_accuracy", metric, epoch + 1)
                # run.track(metric, name='auroc_val', step=epoch+1, context={'subset': 'valid'})
                if (ddp and rank == 0) or (not ddp):
                    run_wandb.log({'auroc_val': metric})
                # run.track(epoch_loss_valid, name='epoch_loss_valid', step=epoch, context={'subset': 'valid'})
                if (ddp and rank == 0) or (not ddp):
                    run_wandb.log({'epoch_loss_valid': epoch_loss_valid})
        # save model
        print(
            f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
        )
        writer.close()

        time_used = datetime.now() - START_TIME
        print(time_used)        
                        
                        
    # inference only
    else:
        model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, val_loader, test_loader)
        # monthdate of trained model
        if loadmodel_path:
            monthdate = loadmodel_path.split("_")[-1].replace(".pth", "")
        else:
            monthdate = datetime.now().strftime("%m%d")
        
        model.eval()
        step = -1
        step_valid = -1
        X_train = []
        X_val = []
        image_paths_train = []
        image_paths_val = []
        y_train = []
        y_val = []
        starting_step_train = 0
            
        # train loader setup if no X_train/X_val and y_train/y_val saved in disk
        if generate_train_features:
            train_iter = iter(train_loader)
            no_crop_train = 0
            # while True:
            for batch_data_idx in tqdm(range(len(train_loader)), total=len(train_loader), ncols=75):
                step += 1
                try:
                    batch_data = next(train_iter)
                except FileNotFoundError:
                        no_crop_train += 1
                        continue
                except EOFError:
                    print(f"Error in step {step}, {image_path}, no cachedataset")
                    continue
                    
                if use_cachedataset:
                    if 'image_path' in batch_data:
                        inputs, labels, image_path = batch_data['image'].to(device), batch_data['label'], batch_data['image_path']
                    else:
                        inputs, labels = batch_data['image'].to(device), batch_data['label']
                        image_path = image_paths_train_[step]
                    if not labels:
                        continue
                else:
                    inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                
                with torch.no_grad():
                    features = model(inputs, return_features=True, inference=True, return_logits=False)
                    if type(features) == tuple and len(features) == 2:
                        features = features[1]
                    features = features.cpu().detach().numpy()
                    features = np.squeeze(features)
                    X_train.append(features)
                    image_paths_train.append(image_path)
                    y_train.append(labels)
            
            
            X_train = np.array(X_train)
            image_paths_train = np.array(image_paths_train)
            y_train = np.array(y_train)
            y_train = np.squeeze(y_train)
            print('no crop train number: ', no_crop_train, 'out of', len(train_loader)) 
            
            if not prop_train:
                with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"X_train_{starting_epoch}epoch_{model_choice}.pkl"), 'wb') as f:
                    pickle.dump(X_train, f)
                with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"image_paths_train_{starting_epoch}epoch_{model_choice}.pkl"), 'wb') as f:
                    pickle.dump(image_paths_train, f)
                with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"y_train_{starting_epoch}epoch_{model_choice}.pkl"), 'wb') as f:
                    pickle.dump(y_train, f) 
             
            # val    
            val_iter = iter(val_loader)
            no_crop_val = 0
            # while True:
            for batch_data_idx in tqdm(range(len(val_loader)), total=len(val_loader), ncols=75):
                step_valid += 1
                try:
                    batch_data = next(val_iter)
                except FileNotFoundError:
                        no_crop_val += 1
                        continue
                
                if use_cachedataset:
                    if 'image_path' in batch_data:
                        inputs, labels, image_path = batch_data['image'].to(device), batch_data['label'], batch_data['image_path']
                    else:
                        inputs, labels = batch_data['image'].to(device), batch_data['label']
                        image_path = image_paths_valid[step_valid]
                else:
                    inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                
                with torch.no_grad():
                    features = model(inputs, return_features=True, inference=True, return_logits=False)
                    if type(features) == tuple and len(features) == 2:
                        features = features[1]
                    features = features.cpu().detach().numpy()
                    features = np.squeeze(features)
                    X_val.append(features)
                    image_paths_val.append(image_path)
                    y_val.append(labels)       
            
            X_val = np.array(X_val)
            image_paths_val = np.array(image_paths_val)
            y_val = np.array(y_val)
            y_val = np.squeeze(y_val)
            print('no crop val number: ', no_crop_val, 'out of', len(val_loader)) 
            
            if not prop_valid:
                with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"X_val_{starting_epoch}epoch_{model_choice}.pkl"), 'wb') as f:
                    pickle.dump(X_val, f)
                with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"image_paths_val_{starting_epoch}epoch_{model_choice}.pkl"), 'wb') as f:
                    pickle.dump(image_paths_val, f)
                with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"y_val_{starting_epoch}epoch_{model_choice}.pkl"), 'wb') as f:
                    pickle.dump(y_val, f) 

                
        # train/val feature from disk
        else:       
            loadmodel_feature = model_choice    
            
            with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"X_train_{starting_epoch}epoch_{loadmodel_feature}.pkl"), 'rb') as f:
                X_train = pickle.load(f)
            with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"image_paths_train_{starting_epoch}epoch_{loadmodel_feature}.pkl"), 'rb') as f:
                image_paths_train = pickle.load(f)
            with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"y_train_{starting_epoch}epoch_{loadmodel_feature}.pkl"), 'rb') as f:
                y_train = pickle.load(f)
            with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"X_val_{starting_epoch}epoch_{loadmodel_feature}.pkl"), 'rb') as f:
                X_val = pickle.load(f)
            with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"image_paths_val_{starting_epoch}epoch_{loadmodel_feature}.pkl"), 'rb') as f:
                image_paths_val = pickle.load(f)
            with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"y_val_{starting_epoch}epoch_{loadmodel_feature}.pkl"), 'rb') as f:
                y_val = pickle.load(f)
                
            if prop_train:
                X_train = X_train[prop_train[0]:prop_train[1]]
                image_paths_train = image_paths_train[prop_train[0]:prop_train[1]]
                y_train = y_train[prop_train[0]:prop_train[1]]

        # valid loader
        if False:
            labels = []
            X_val = []
            y_val = []
            for val_data in tqdm(val_loader):
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                with torch.no_grad():
                    features = model(val_images, inference=True)
                    X_val.append(features.cpu().detach().numpy())
                    y_val.append(val_labels.cpu().detach().numpy())

            X_val = np.concatenate(X_val, axis=0)
            y_val = np.concatenate(y_val, axis=0)
            
            y_train_proba = linear_model.predict_proba(X_train)[::, 1]
            y_val_proba = linear_model.predict_proba(X_val)[::, 1]
            metric = run_analysis("Logistic Regression", y_train, y_train_proba, y_val, y_val_proba)
            
            # save linear model
            month_date = datetime.now().strftime("%m%d")
            linear_model_save_path = os.path.join(model_save_path, f"linear_model_{starting_epoch}epoch_{model_choice}_{month_date}.pkl")
            with open(linear_model_save_path,'wb') as f:
                pickle.dump(linear_model, f)
         
        start_linearprobe_time = datetime.now()   
         
        testset_large_volume = 0
        no_crop_test = 0 
        
        X_test_path = os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"X_test_{starting_epoch}epoch_{model_choice}.pkl")
        image_paths_test_path = os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"image_paths_test_{starting_epoch}epoch_{model_choice}.pkl")
        y_test_path = os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"y_test_{starting_epoch}epoch_{model_choice}.pkl")
        
        # calculate test features, if not in disk 
        if not os.path.exists(X_test_path) or not os.path.exists(image_paths_test_path) or not os.path.exists(y_test_path):
            X_test = []
            image_paths_test = []
            y_test = []

            step_test = -1
            time_bin_0_is_event = []
            time_bin_0_log_time = []
            is_event_ls = []
            log_time_ls = []
            time_ls = []
            time_dependent_logits_ls = []
            
            test_loader = test_loader # test_loader
            print('================', f"{'val_loader' if test_loader == val_loader else 'test_loader'}", 'for testing len:', len(test_loader), '----------')
            print()
            test_iter = iter(test_loader)
            while True:
                try:
                    step_test += 1
                    if step_test % 100 == 0:
                        print(f"Current step: {step_test}, time used: {datetime.now() - START_TIME}")
                    
                    try:
                        batch_data = next(test_iter)
                    except FileNotFoundError:
                        no_crop_test += 1
                        continue

                    if not use_cachedataset:
                        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                    else:
                        if 'image_path' in batch_data:
                            inputs, labels, image_path  = batch_data['image'].to(device), batch_data['label'], batch_data['image_path']
                        else:
                            inputs, labels = batch_data['image'].to(device), batch_data['label']
                            image_path = image_paths_test_[step_test]
                    
                    # collect test set features
                    with torch.no_grad():
                        try:
                            y_pred, time_independent_features = model(inputs, return_features=True, inference=inference, return_logits=True) 
                            
                        # RuntimeError: GET was unable to find an engine to execute this computation
                        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                            testset_large_volume += 1
                            print(e, 'testset_large_volume:', testset_large_volume)
                            continue
                    features = time_independent_features.cpu().detach().numpy()
                    y_pred = y_pred.cpu().detach().numpy()

                    features = np.squeeze(features)
                    X_test.append(features)
                    image_paths_test.append(image_path)
                    y_test.append(labels)
                    
                    if 'RSNA' not in dataformat and 'chexpert' not in dataformat:
                        # time dependent c statistics
                        time_dependent_logits_ls.append(y_pred)
                        final_batch, motor_batch = get_final_batch(image_path, nii_folder, motor_task, test_dataset, index_test, ontology)  
                        motor_batch['censor_time'] 
                        is_event = final_batch["is_event"]
                        log_time = final_batch["log_time"]
                        time_ = final_batch["time"]
                        is_event_ls.append(is_event.cpu().detach().numpy())
                        log_time_ls.append(log_time.cpu().detach().numpy())
                        time_ls.append(time_.cpu().detach().numpy())
                    
                except StopIteration:
                    print(f"StopIteration at step {step_test}")
                    break        
            # save test features
            if loadmodel_path:
                path_ls = loadmodel_path.split('_')
                for chunk in path_ls:
                    if 'epoch' in chunk:
                        try:
                            starting_epoch = int(chunk.replace('epoch', ''))  
                        except:
                            starting_epoch = -1
                        break
            else:
                starting_epoch = -1

            X_test = np.array(X_test)

            if not prop_test:
                with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"X_test_{starting_epoch}epoch_{model_choice}.pkl"), 'wb') as f:
                    pickle.dump(X_test, f)
                with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"image_paths_test_{starting_epoch}epoch_{model_choice}.pkl"), 'wb') as f:
                    pickle.dump(image_paths_test, f)
                with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"y_test_{starting_epoch}epoch_{model_choice}.pkl"), 'wb') as f:
                    pickle.dump(y_test, f) 
                
            if 'RSNA' not in dataformat:
                # time dependent c statistics    
                with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"time_dependent_logits_ls_{starting_epoch}epoch_{model_choice}.pkl"), 'wb') as f:
                    pickle.dump(time_dependent_logits_ls, f) 
                with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"is_event_ls_{starting_epoch}epoch_{model_choice}.pkl"), 'wb') as f:
                    pickle.dump(is_event_ls, f) 
                with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"log_time_ls_{starting_epoch}epoch_{model_choice}.pkl"), 'wb') as f:
                    pickle.dump(log_time_ls, f) 
                with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"time_ls_{starting_epoch}epoch_{model_choice}.pkl"), 'wb') as f:
                    pickle.dump(time_ls, f) 
            
        # load test features from disk
        else:  
            with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"X_test_{starting_epoch}epoch_{model_choice}.pkl"), 'rb') as f:
                X_test = pickle.load(f)
                X_test = np.array(X_test) 
            with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"image_paths_test_{starting_epoch}epoch_{model_choice}.pkl"), 'rb') as f:
                image_paths_test = pickle.load(f)
            with open(os.path.join(model_save_path.replace('model_checkpoints', 'features'), f"y_test_{starting_epoch}epoch_{model_choice}.pkl"), 'rb') as f:
                y_test = pickle.load(f) 


        ## linear probe 1) logistic regression 2) survival probe
        # train logistic regression model for each task on the spot instead of load LR from disk
        if False:
            # load linear model weights
            month_date = monthdate
            linear_model_save_path = os.path.join(model_save_path, f"linear_model_{starting_epoch}epoch_{model_choice}_{month_date}.pkl")
            with open(linear_model_save_path, 'rb') as f:
                linear_model = pickle.load(f)
            y_test_proba = linear_model.predict_proba(X_test)[::, 1]
            metric = run_analysis("Logistic Regression", y_test, y_test_proba)
        else:
            # train linear model
            if loadmodel_path:
                path_ls = loadmodel_path.split('_')
                for chunk in path_ls:
                    if 'epoch' in chunk:
                        epoch = int(chunk.replace('epoch', ''))  
                        break
            else:
                epoch = -1
            month_date = monthdate
            
            if label_column:
                if 'chexpert' in dataformat:
                    X_train = np.concatenate((X_train, X_val), axis=0)
                    image_paths_train = np.concatenate((image_paths_train, image_paths_val), axis=0)
                    y_train = np.concatenate((y_train, y_val), axis=0)
                    X_test = np.concatenate((X_test, X_val), axis=0)
                    image_paths_test = np.concatenate((image_paths_test, image_paths_val), axis=0)
                    y_test = np.concatenate((y_test, y_val), axis=0)
                else:
                    X_train = np.concatenate((X_train, X_val), axis=0)
                    image_paths_train = np.concatenate((image_paths_train, image_paths_val), axis=0)
                    y_train = np.concatenate((y_train, y_val), axis=0)
                
                metric_values, _, _, auroc_val, auroc_train_dict, auroc_val_dict = logistic_regression(label_column, X_train, y_train , X_test, y_test, model_save_path, model_choice, epoch, month_date, metric_values, None, None, tune_linearprobe)
                for key, value in auroc_val_dict.items():
                    # run.track(value, name=key, step=epoch, context={'subset': 'test'})
                    if (ddp and rank == 0) or (not ddp):
                        try:
                            run_wandb.log({key: value})
                            writer.add_scalar(key, value, epoch)
                        except:
                            print('error in logging:', key, value)
                            pass

            if survival_tasks:
                tdcs_train_dict, tdcs_test_dict, ibs_train_dict, ibs_test_dict = survival_probe(survival_tasks, X_train, image_paths_train, X_val, image_paths_val, X_test, image_paths_test, label_df_tte, nii_folder)
                for key, value in tdcs_train_dict.items():
                    # run.track(value, name=key, step=epoch, context={'subset': 'train'})
                    if (ddp and rank == 0) or (not ddp):
                        run_wandb.log({key: value})
                    try:
                        writer.add_scalar(key, value, epoch)
                    except:
                        print('error in logging:', key, value)
                        pass
                for key, value in tdcs_test_dict.items():
                    # run.track(value, name=key, step=epoch, context={'subset': 'test'})
                    if (ddp and rank == 0) or (not ddp):
                        run_wandb.log({key: value})
                    try:
                        writer.add_scalar(key, value, epoch)
                    except:
                        print('error in logging:', key, value)
                        pass
                for key, value in ibs_train_dict.items():
                    # run.track(value, name=key, step=epoch, context={'subset': 'train'})
                    if (ddp and rank == 0) or (not ddp):
                        run_wandb.log({key: value})
                    try:                    
                        writer.add_scalar(key, value, epoch)
                    except:
                        print('error in logging:', key, value)
                        pass
                for key, value in ibs_test_dict.items():
                    # run.track(value, name=key, step=epoch, context={'subset': 'test'})
                    if (ddp and rank == 0) or (not ddp):
                        run_wandb.log({key: value})
                    try:
                        writer.add_scalar(key, value, epoch)
                    except:
                        print('error in logging:', key, value)
                        pass
                    
        print("Test set large volume that can't fit into GPU: ", testset_large_volume)
        print('no crop test number: ', no_crop_test, 'out of', len(test_loader))
        writer.close()
        print('linear probe time used: ', datetime.now() - start_linearprobe_time)
        time_used = datetime.now() - START_TIME
        print(time_used)
    
    if (ddp and rank == 0) or (not ddp):
        run_wandb.finish()    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image model")
    parser.add_argument(
        "--dataformat",
        type=str,
        help="data intake format, e.g. 'nii_gz'",
        default="nii_gz",
    )
    parser.add_argument(
        "--label_csv",
        type=str,
        help="path to the csv file containing the labels",
        default="label.csv",
    )
    parser.add_argument(
        "--CT_8192labels",
        type=str,
        help="path to the pickle file containing the same visit 8192 labels",
        default="CT_8192labels.pkl",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        help="path to the csv file containing the labels",
        default="model_checkpoints",
    )
    parser.add_argument(
        "--loadmodel_path",
        type=str,
        help="path to the csv file containing the labels",
        default=None,
    )
    parser.add_argument(
        "--model_choice",
        type=str,
        help="model choice, e.g. 'unet', 'densenet', 'swin'",
        default="unet",
    )
    parser.add_argument(
        "--finetune_label",
        type=str,
        help="label choice, e.g. '12_month_PH'",
        default="12_month_PH",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        nargs="*",
        help="label choice, e.g. '12_month_PH' or a list of labels ['12_month_PH', 'PE']",
        default=[],
    )
    parser.add_argument(
        "--survival_tasks",
        type=str,
        nargs="*",
        help="survival label choice, e.g. 'mortality' or a list of labels ['mortality', 'Edema']",
        default=[],
    )
    parser.add_argument(
        "--prop_train",
        type=int,
        nargs="*",
        help="how many train data instances to use",
        default=[],
    )
    parser.add_argument(
        "--prop_valid",
        type=int,
        nargs="*",
        help="how many valid data instances to use",
        default=[],
    )
    parser.add_argument(
        "--prop_test",
        type=int,
        nargs="*",
        help="how many test data instances to use",
        default=[],
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        help="how many epochs to wait before validation",
        default=1,
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        help="how many epochs to train for",
        default=5,
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        help="how many steps to accumulate gradients for",
        default=8,
    )
    parser.add_argument(
        "--linear_probe",
        action='store_true',
        help="If specified, only train the last layer of the model.",
        default=False,
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        help="vocab size for the tokenizer",
        default=65536,
    )
    parser.add_argument(
        "--from_pretrained_tokenizer",
        action='store_true',
        help="If specified, only train the last layer of the model.",
        default=False,
    )
    parser.add_argument(
        "--nii_folder",
        type=str,
        help="nii folder path",
        default="anon_nii_gz",
    )
    parser.add_argument(
        "--inference",
        action='store_true',
        help="If specified, only do inference",
        default=False,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="batch size for train, val, test loaders",
        default=4,
    )
    parser.add_argument(
        "--use_cachedataset",
        action='store_true',
        help="If specified, use cached dataset",
        default=False,
    )
    parser.add_argument(
        "--parquet_folder",
        type=str,
        help="nii folder path",
        default="timelines_smallfiles_meds",
    )
    parser.add_argument(
        "--TARGET_DIR",
        type=str,
        help="nii folder path",
        default='training/trash',
    )
    parser.add_argument(
        "--ontology_path",
        type=str,
        help="path to the ontology file",
        default="inspect_ontology.pkl",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        help="batch size for train, val, test loaders",
        default=20,
    )
    parser.add_argument(
        "--month_date_hour",
        type=str,
        help="month_date_hour to load model/tokenizer/motortask",
        default=None,
    )
    parser.add_argument(
        "--only_train_tokenizer",
        action='store_true',
        help="If specified, only train the tokenizer.",
        default=False,
    )
    parser.add_argument(
        "--final_layer_size",
        type=int,
        help="batch size for train, val, test loaders",
        default=512,
    )
    parser.add_argument(
        "--mixed_precision",
        action='store_true',
        help="If specified, use mixed precision training.",
        default=False,
    )
    parser.add_argument(
        "--use_checkpoint",
        action='store_true',
        help="If specified, use_checkpoint",
        default=False,
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        help="num TTE tasks for pretraining",
        default=8192,
    )
    parser.add_argument(
        "--frozen_vocab_layer",
        action='store_true',
        help="If specified, freeze the vocab layer.",
        default=False,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="learning rate for the optimizer",
        default=1e-4,
    )
    parser.add_argument(
        "--dropout_prob",
        type=float,
        help="dropout probability for the model",
        default=0.0,
    )
    parser.add_argument(
        "--test_subset",
        action='store_true',
        help="If specified, use a subset of the test/valid data.",
        default=False,
    )
    parser.add_argument(
        "--pretrained_path_swinUNETR",
        type=str,
        help="path to the SwinUNETR pretrained weights",
        default="/local-scratch/nigam/users/zphuo/model_weights/ssl_pretrained_weights.pth",
    )
    parser.add_argument(
        "--pretrained_path_densenet",
        type=str,
        help="path to the densenet pretrained weights",
        default="/local-scratch/nigam/users/zphuo/model_weights/i3densenet.pth",
    )
    parser.add_argument(
        "--pretrained_path_resnet",
        type=str,
        help="path to the densenet pretrained weights",
        default="/local-scratch/nigam/users/zphuo/model_weights/i3resnet.pth",
    )
    parser.add_argument(
        "--ddp",
        action='store_true',
        help="If specified, use ddp for parallelize multi GPU.",
        default=False,
    )
    parser.add_argument(
        "--tune_linearprobe",
        action='store_true',
        help="If specified, use tune linear probe",
        default=False,
    )
    parser.add_argument(
        "--use_crop",
        action='store_true',
        help="If specified, use cropped CT",
        default=False,
    )
    parser.add_argument(
        "--unet_out_channels",
        type=int,
        help="batch size for train, val, test loaders",
        default=384,
    )
    parser.add_argument(
        "--multitask",
        action='store_true',
        help="If specified, multitask direct supervision for continued pretraining",
        default=False,
    )
    parser.add_argument(
        "--aim_hash",
        type=str,
        help="aim hash",
        default=None,
    )
    parser.add_argument(
        "--generate_train_features",
        action='store_true',
        help="If specified, generate_train_features",
        default=False,
    )
    parser.add_argument(
        "--wandb_run_id",
        type=str,
        help="wandb_run_id",
        default=None,
    )
    parser.add_argument(
        "--wandb_group_name",
        type=str,
        help="wandb_group_name",
        default='None',
    )
    
    args = parser.parse_args()
    train(
        args,
        args.dataformat,
        args.label_csv,
        args.CT_8192labels,
        args.model_save_path,
        args.loadmodel_path,
        args.model_choice,
        args.prop_train,
        args.prop_valid,
        args.prop_test,
        args.finetune_label,
        args.label_column,
        args.survival_tasks,
        args.val_interval,
        args.max_epochs,
        args.linear_probe,
        args.vocab_size,
        args.from_pretrained_tokenizer,
        args.nii_folder,
        args.inference,
        args.batch_size,
        args.accumulation_steps,
        args.use_cachedataset,
        args.parquet_folder,
        args.TARGET_DIR,
        args.ontology_path,
        args.num_proc,
        args.month_date_hour,
        args.only_train_tokenizer,
        args.final_layer_size,
        args.mixed_precision,
        args.use_checkpoint,
        args.num_tasks,
        args.frozen_vocab_layer,
        args.learning_rate, 
        args.dropout_prob,
        args.test_subset,
        args.pretrained_path_swinUNETR,
        args.pretrained_path_densenet,
        args.pretrained_path_resnet,
        args.ddp,
        args.unet_out_channels,
        args.tune_linearprobe,
        args.use_crop,
        args.multitask,
        args.aim_hash,
        args.generate_train_features,
        args.wandb_run_id,
        args.wandb_group_name,
    )
