from datetime import datetime
START_TIME = datetime.now()

print("Start Time: ", START_TIME)

import logging
import os
import sys
import shutil
import tempfile

import matplotlib.pyplot as plt
import torch
import pydicom
import tarfile
import glob
import pandas as pd
import argparse
import wandb
from tqdm import tqdm
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict
from torch.utils.data import DataLoader, DistributedSampler

from torch.utils.tensorboard import SummaryWriter
import numpy as np
from monai.networks.nets import SwinUNETR
from sklearn.metrics import roc_auc_score

import monai
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import DataLoader, ImageDataset, PersistentDataset
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

from networks import DenseNet121
from network_louis import SwinClassifier
from utils import (
    load_pretrained_swinunetr,
    SwinUNETRForClassification,
    DicomDataset,
    TarImageDataset,
    CustomToOneChannel,
    CustomToOneChanneld
)

# wandb.init(project="TTE", entity="stanford_som")

pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
print('available gpus: ', torch.cuda.device_count())

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
print_config()

# Set data directory

root_dir = "/share/pi/nigam/projects/zphuo/data/PE/Jose_monai_MRI"
print(root_dir)

time_used = datetime.now() - START_TIME
print(time_used)



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
def cleanup():
    dist.destroy_process_group()

def train(
    dataformat,
    label_csv,
    model_save_path,
    loadmodel_path,
    model_choice,
    prop_train,
    prop_val,
    label_column,
    val_interval,
    max_epochs,
    linear_probe,
    batch_size,
    nii_folder,
    use_cachedataset,
):
    
    
    # Define transforms
    if use_cachedataset:
        train_transforms = Compose(
            [LoadImaged(keys=["image"]), ScaleIntensityd(keys=["image"]), EnsureChannelFirstd(keys=["image"]), Resized(keys=["image"], spatial_size=(224, 224, 224)), 
            RandRotate90d(keys=["image"]), 
            CustomToOneChanneld(keys=["image"])]
        )
        val_transforms = Compose(
            [LoadImaged(keys=["image"]), ScaleIntensityd(keys=["image"]), EnsureChannelFirstd(keys=["image"]), Resized(keys=["image"], spatial_size=(224, 224, 224)), CustomToOneChanneld(keys=["image"])]
        )
        
    else:    
        train_transforms = Compose(
            [ScaleIntensity(), EnsureChannelFirst(), Resize((224, 224, 224)), RandRotate90(), CustomToOneChannel()]
        )

        val_transforms = Compose(
            [ScaleIntensity(), EnsureChannelFirst(), Resize((224, 224, 224)), CustomToOneChannel()]
        )
    # rank = int(os.environ['RANK'])
    # world_size = int(os.environ['WORLD_SIZE'])
    # setup(rank, world_size)
    # device = torch.device(f'cuda:{rank}')
    
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

        label_df = pd.read_csv(label_csv)
        label_column = label_column

        # label column unique values, top as 1, second top as 0
        label_dict = label_df[label_column].value_counts().to_dict()
        keys = list(label_dict.keys())
        if 'True' in keys and 'False' in keys:
            pos_key = 'True'
            neg_key = 'False'
        else:
            pos_key = keys[0]
            neg_key = keys[1]

        labels = []
        image_paths = []
        for idx, row in label_df.iterrows():
            labels.append(1 if row[label_column] == pos_key else 0)
            image_paths.append(
                nii_folder
                + "/"
                + str(row["patient_id"])
                + "_"
                + row["procedure_time"].replace(":", "_").replace("T", "_")
                + ".nii.gz"
            )

        train_idx = label_df["split"] == "train"

        labels_train = np.array(labels)[train_idx]
        print(np.unique(labels_train, return_counts=True), 'labels_train')
        image_paths_train = np.array(image_paths)[train_idx]

        if prop_train:
            image_paths_train = image_paths_train[:prop_train]
        
        # full val set
        label_csv_full = label_csv.replace('_subset', '')
        label_df_full = pd.read_csv(label_csv_full)
        labels = []
        image_paths = []
        for idx, row in label_df_full.iterrows():
            labels.append(1 if row[label_column] == pos_key else 0)
            image_paths.append(
                nii_folder
                + "/"
                + str(row["patient_id"])
                + "_"
                + row["procedure_time"].replace(":", "_").replace("T", "_")
                + ".nii.gz"
            )
        val_idx = label_df_full["split"] == "valid"
        labels_valid = np.array(labels)[val_idx]
        print(np.unique(labels_valid, return_counts=True), 'labels_valid')
        image_paths_valid = np.array(image_paths)[val_idx]
        
        if prop_val:
            image_paths_valid = image_paths_valid[:prop_val]

        # create a training data loader
        
        if use_cachedataset:
            data_train = []
            for i in range(len(image_paths_train)):
                one_entry = {'image': image_paths_train[i], 'label': labels_train[i]}
                data_train.append(one_entry)
            data_val = []
            for i in range(len(image_paths_valid)):
                one_entry = {'image': image_paths_valid[i], 'label': labels_valid[i]}
                data_val.append(one_entry)
            train_ds = PersistentDataset(
                data=data_train,
                transform=train_transforms,
                #cache_num=9223,
                cache_dir=os.path.join(model_save_path, 'cache_dir'),
            )
            val_ds = PersistentDataset(
                data=data_val,
                transform=val_transforms,
                #cache_num=9223,
                cache_dir=os.path.join(model_save_path, 'cache_dir'),
            ) 
                
        else:
            train_ds = ImageDataset(
                image_files=image_paths_train,
                labels=labels_train,
                transform=train_transforms,
            )
            # train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank)

            # create a validation data loader
            val_ds = ImageDataset(
                image_files=image_paths_valid,
                labels=labels_valid,
                transform=val_transforms,
            )
            
            
        train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin_memory,
                # sampler = train_sampler,
            )
        val_loader = DataLoader(
                val_ds, batch_size=batch_size, num_workers=4, pin_memory=pin_memory
            )

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
        
    if model_choice == "unet":
        # Define your SwinUNETR parameters in a dictionary
        swin_unetr_params = {
            "img_size": (224, 224, 224),
            "in_channels": 1,
            "out_channels": 2,  # Used for segmentation, but will be adapted
            "feature_size": 48,
            "drop_rate": 0.0,
            "attn_drop_rate": 0.0,
            "dropout_path_rate": 0.0,
            "use_checkpoint": True,
        }

        # Initialize the SwinUNETRForClassification model
        model = SwinUNETRForClassification(
            swin_unetr_params,
            num_classes=2,  # Specify the number of classes for classification
        ).to(device)
            
        if loadmodel_path:
            state_dict = torch.load(loadmodel_path)
            model.load_state_dict(state_dict)
        else:
            pretrained_path = "/share/pi/nigam/projects/zphuo/data/PE/Jose_monai_MRI/ssl_pretrained_weights.pth"
            model = load_pretrained_swinunetr(
                model, use_pretrained=True, pretrained_path=pretrained_path
            )
            
        if linear_probe:
            print("Linear probe for SwinUNETR...")
            # Freeze all layers in the model
            for name, param in model.named_parameters():
                if "out" not in name:
                    param.requires_grad = False
                    
        # ddp_model = DDP(model, device_ids=[rank])
        
        
    if model_choice == "unet_louis":
        # Define your SwinUNETR parameters in a dictionary
        swin_unetr_params = {
            #"img_size": (224, 224, 224),
            "in_channels": 1,
            "out_channels": 2,  # Used for segmentation, but will be adapted
            "feature_size": 48,
            "drop_rate": 0.0,
            "attn_drop_rate": 0.0,
            "dropout_path_rate": 0.0,
            "use_checkpoint": True,
        }

        if loadmodel_path:
            # Initialize the SwinUNETRForClassification model
            model = SwinClassifier(
                **swin_unetr_params,
                #num_classes=2,  # Specify the number of classes for classification
            ).to(device)
            state_dict = torch.load(loadmodel_path)
            model.load_state_dict(state_dict)
        else:
            pretrained_path = "/share/pi/nigam/projects/zphuo/data/PE/Jose_monai_MRI/ssl_pretrained_weights.pth"
            # model = load_pretrained_swinunetr(
            #     model, use_pretrained=True, pretrained_path=pretrained_path
            # )
            model = SwinClassifier(
                **swin_unetr_params,
                pretrained_path=pretrained_path
            )
            
        if linear_probe:
            print("Linear probe for SwinUNETR...")
            # Freeze all layers in the model
            for name, param in model.named_parameters():
                if "out" not in name:
                    param.requires_grad = False
                    
        
    elif model_choice == "densenet":
        model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
        if loadmodel_path:
            state_dict = torch.load(loadmodel_path)
            model.load_state_dict(state_dict)
          
        if linear_probe:
            print("Linear probe for densenet...")
            # Freeze all layers in the model
            for name, param in model.named_parameters():
                if "out" not in name:
                    param.requires_grad = False
    elif model_choice == "swin":
        model = SwinClassifier(spatial_dims=3, in_channels=1, out_channels=2).to(device)
        if loadmodel_path:
            state_dict = torch.load(loadmodel_path)
            model.load_state_dict(state_dict)

    # CrossEntropyLoss and Adam optimizer
    loss_function = torch.nn.CrossEntropyLoss()
    # loss_function = torch.nn.BCEWithLogitsLoss()  # also works with this data

    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    # start a typical PyTorch training
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    writer = SummaryWriter()

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in tqdm(train_loader):
            step += 1
            
            if use_cachedataset:
                inputs, labels = batch_data['image'].to(device), batch_data['label'].to(device)
            else:
                inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            # import pdb;pdb.set_trace()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            time_used = datetime.now() - START_TIME
            # print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}, time_used: {time_used}", end="  ")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()

            num_correct = 0.0
            metric_count = 0
            proba = []
            labels = []
            for val_data in tqdm(val_loader):
                if use_cachedataset:
                    val_images, val_labels = val_data['image'].to(device), val_data['label'].to(device)
                else:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)

                with torch.no_grad():
                    val_outputs = model(val_images)
                    # value = torch.eq(val_outputs.argmax(dim=1), val_labels.argmax(dim=1))
                    value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                    metric_count += len(value)
                    num_correct += value.sum().item()
                    #proba.extend(val_outputs.argmax(dim=1).cpu())
                    proba.extend(val_outputs[:,1].cpu())
                    labels.extend(val_labels.cpu())

            # metric = num_correct / metric_count
            metric = roc_auc_score(labels, proba)
            metric_values.append(metric)

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                
                torch.save(
                    model.state_dict(), os.path.join(model_save_path, f"best_metric_model_{epoch}epoch_{model_choice}{'_lb' if linear_probe else ''}.pth")
                )

            print(f"Current epoch: {epoch+1} current accuracy: {metric:.4f} ")
            print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
            writer.add_scalar("val_accuracy", metric, epoch + 1)

    # save model
    
    print(
        f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
    )
    writer.close()

    time_used = datetime.now() - START_TIME
    print(time_used)


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
        default="/share/pi/nigam/projects/zphuo/data/omop_extract_PHI/som-nero-phi-nigam-starr.frazier/cohort_0.2.0_master_file_anon.csv",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        help="path to the csv file containing the labels",
        default="/share/pi/nigam/projects/zphuo/data/PE/Jose_monai_MRI/model_checkpoints",
    )
    parser.add_argument(
        "--loadmodel_path",
        type=str,
        help="load model from this path",
        default=None,
    )
    parser.add_argument(
        "--model_choice",
        type=str,
        help="model choice, e.g. 'unet', 'densenet', 'swin'",
        default="unet",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        help="label choice, e.g. '12_month_PH'",
        default="12_month_PH",
    )
    parser.add_argument(
        "--prop_train",
        type=int,
        help="how many train data instances to use",
        default=None,
    )
    parser.add_argument(
        "--prop_valid",
        type=int,
        help="how many valid data instances to use",
        default=None,
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        help="how many epochs to wait before validation",
        default=None,
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        help="how many epochs to train for",
        default=5,
    )
    parser.add_argument(
        "--linear_probe",
        action='store_true',
        help="If specified, only train the last layer of the model.",
        default=False,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="batch size for train, val, test loaders",
        default=4,
    )
    parser.add_argument(
        "--nii_folder",
        type=str,
        help="nii folder path",
        default="/share/pi/nigam/data/inspect/anon_nii_gz",
    )
    parser.add_argument(
        "--use_cachedataset",
        action='store_true',
        help="If specified, use cached dataset",
        default=False,
    )
    args = parser.parse_args()
    train(
        args.dataformat,
        args.label_csv,
        args.model_save_path,
        args.loadmodel_path,
        args.model_choice,
        args.prop_train,
        args.prop_valid,
        args.label_column,
        args.val_interval,
        args.max_epochs,
        args.linear_probe,
        args.batch_size,
        args.nii_folder,
        args.use_cachedataset,
    )
