import argparse
import logging
import os
import sys
import shutil
import tempfile
import matplotlib.pyplot as plt
import torch
import numpy as np
import monai
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import DataLoader, ImageDataset
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
)
from monai.networks.nets import DenseNet121
from pe.datatasets.data_loaders.rsna_pe_simple import NiftiDataset
from torch.utils.data import DataLoader, Subset, random_split

def main(args):
    csv_file = args.csv_file
    root_dir = args.root_dir

    pin_memory = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print_config()

    # Define transforms
    train_transforms = Compose([ScaleIntensity(), Resize((224, 224, 200)), RandRotate90()])

    #nifti_dataset = NiftiDataset(csv_file=csv_file, root_dir=root_dir,transform=train_transforms)
    nifti_dataset = NiftiDataset(csv_file=csv_file, root_dir=root_dir)
    loader        = DataLoader(nifti_dataset, batch_size=1, shuffle=True)

    dataset_size = len(loader.dataset)
    test_size    = int(0.2 * dataset_size)
    train_size   = dataset_size - test_size

    # Create indices for train and test subsets
    indices = list(range(dataset_size))
    train_indices, test_indices = indices[:train_size], indices[train_size:]

    # Create Subset data loaders for train and test
    train_loader = DataLoader(Subset(loader.dataset, train_indices), batch_size=1, shuffle=True)
    test_loader  = DataLoader(Subset(loader.dataset, test_indices), batch_size=1, shuffle=False)

    # Create a model (Swin or DenseNet)
    model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=1).to(device)

    # Create loss function and optimizer
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer     = torch.optim.Adam(model.parameters(), 1e-6)


        # start a typical PyTorch training

    val_interval      = 2
    best_metric       = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values     = []
    max_epochs        = 5

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            inputs         = torch.unsqueeze(train_transforms(inputs),dim=0)
            optimizer.zero_grad()
            # import pdb;pdb.set_trace()
            outputs  = model(inputs)
            loss     = loss_function(outputs,  labels.reshape_as(outputs).float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len   = train_size // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()

            num_correct = 0.0
            metric_count = 0
            for val_data in test_loader:
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                with torch.no_grad():
                    val_outputs = model(val_images)
                    value = torch.eq(val_outputs.argmax(dim=1), val_labels.argmax(dim=1))
                    metric_count += len(value)
                    num_correct += value.sum().item()

            metric = num_correct / metric_count
            metric_values.append(metric)

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_metric_model_classification3d_array.pth")
                print("saved new best metric model")

            print(f"Current epoch: {epoch+1} current accuracy: {metric:.4f} ")
            print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
            

    print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV file and root directory.")
    parser.add_argument("--csv_file", type=str, default='/share/pi/nigam/data/RSNAPE/simplified_labels/train.csv')
    parser.add_argument("--root_dir", type=str, default='/share/pi/nigam/data/RSNAPE/nifti_crop/')

    args = parser.parse_args()
    main(args)
