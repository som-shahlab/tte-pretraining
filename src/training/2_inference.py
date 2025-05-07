#!/usr/bin/env python
# coding: utf-8

from networks import DenseNet121_TTE


import torch
import pandas as pd
import numpy as np
from monai.data import DataLoader, ImageDataset
import monai
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
)

from torch.utils.tensorboard import SummaryWriter

from utils import (
    load_pretrained_swinunetr,
    SwinUNETRForClassification,
    DicomDataset,
    TarImageDataset,
    CustomToOneChannel,
    run_analysis,
)

from_pretrained_tokenizer = True
vocab_size = 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = torch.cuda.is_available()

model_choice = "densenet_tte"
loadmodel_path = f"tte_models/best_metric_model_3epoch_{model_choice}.pth"
label_column = "12_month_PH"
nii_folder = "anon_nii_gz"


###### set up MOTOR task ########
################################

import shutil
import os


TARGET_DIR = "trash/tutorial_6_INSEPCT"

from_pretrained = from_pretrained_tokenizer
num_proc = 20

if not from_pretrained:
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)

    os.mkdir(TARGET_DIR)
    os.mkdir(os.path.join(TARGET_DIR, "motor_model"))

import datasets
import femr.index
import femr.splits

# First, we want to split our dataset into train, valid, and test
# We do this by calling our split functionality twice

# dataset = datasets.Dataset.from_parquet('input/meds/data/*')
parquet_folder = "timelines_smallfiles_meds/data_subset/*"
dataset = datasets.Dataset.from_parquet(parquet_folder)


index = femr.index.PatientIndex(dataset, num_proc=num_proc)
main_split = femr.splits.generate_hash_split(
    index.get_patient_ids(), 97, frac_test=0.15
)


# Note that we want to save this to the target directory since this is important information

main_split.save_to_csv(os.path.join(TARGET_DIR, "motor_model", "main_split.csv"))

train_split = femr.splits.generate_hash_split(
    main_split.train_patient_ids, 87, frac_test=0
)

# print(train_split.train_patient_ids)
# print(train_split.test_patient_ids)

main_dataset = main_split.split_dataset(dataset, index)
train_dataset = train_split.split_dataset(
    main_dataset["train"],
    femr.index.PatientIndex(main_dataset["train"], num_proc=num_proc),
)

import femr.models.tokenizer
from femr.models.tokenizer import FEMRTokenizer
import pickle

# First, we need to train a tokenizer
# Note, we need to use a hierarchical tokenizer for MOTOR


with open("ontology.pkl", "rb") as f:
    ontology = pickle.load(f)
if not from_pretrained:
    tokenizer = femr.models.tokenizer.train_tokenizer(
        main_dataset["train"],
        vocab_size=vocab_size,
        is_hierarchical=True,
        num_proc=num_proc,
        ontology=ontology,
    )

    # Save the tokenizer to the same directory as the model
    tokenizer.save_pretrained(os.path.join(TARGET_DIR, "motor_model"))

else:
    # load pretrained tokenizer
    tokenizer = femr.models.tokenizer.FEMRTokenizer.from_pretrained(
        os.path.join(TARGET_DIR, "motor_model"), ontology=ontology
    )
import femr.models.tasks

if "subset" in parquet_folder:
    num_tasks = 39
else:
    num_tasks = 64

# Second, we need to prefit the MOTOR model. This is necessary because piecewise exponential models are unstable without an initial fit

motor_task = femr.models.tasks.MOTORTask.fit_pretraining_task_info(
    main_dataset["train"],
    tokenizer,
    num_tasks=num_tasks,
    num_bins=4,
    final_layer_size=32,
    num_proc=num_proc,
)


import femr.models.processor
import femr.models.tasks

# Third, we need to create batches.

processor = femr.models.processor.FEMRBatchProcessor(tokenizer, motor_task)


index_train = femr.index.PatientIndex(train_dataset["train"], num_proc=num_proc)
# We can do this one patient at a time


transformer_config = femr.models.transformer.FEMRTransformerConfig(
    vocab_size=tokenizer.vocab_size,
    is_hierarchical=tokenizer.is_hierarchical,
    n_layers=2,
    hidden_size=764,
    intermediate_size=64 * 2,
    n_heads=8,
)

config = femr.models.transformer.FEMRModelConfig.from_transformer_task_configs(
    transformer_config, motor_task.get_task_config()
)


###### set up MOTOR task ########
################################


model = DenseNet121_TTE(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    time_bins=motor_task.time_bins,
    pretraining_task_info=motor_task.get_task_config().task_kwargs[
        "pretraining_task_info"
    ],
    final_layer_size=motor_task.final_layer_size,
    vocab_size=tokenizer.vocab_size,
    device=device,
).to(device)


if loadmodel_path:
    state_dict = torch.load(loadmodel_path)
    model.load_state_dict(state_dict)


label_csv = "labels.csv"


label_csv = pd.read_csv(label_csv)
label_column = label_column

labels = []
image_paths = []
for idx, row in label_csv.iterrows():
    labels.append(1 if row[label_column] == "True" else 0)
    image_paths.append(
        nii_folder
        + "/"
        + str(row["patient_id"])
        + "_"
        + row["procedure_time"].replace(":", "_").replace("T", "_")
        + ".nii.gz"
    )

train_idx = label_csv["split"] == "train"
valid_idx = label_csv["split"] == "valid"
test_idx = label_csv["split"] == "test"

labels_train = np.array(labels)[train_idx]
labels_valid = np.array(labels)[valid_idx]
labels_test = np.array(labels)[test_idx]

image_paths_train = np.array(image_paths)[train_idx]
image_paths_valid = np.array(image_paths)[valid_idx]
image_paths_test = np.array(image_paths)[test_idx]


# Define transforms
train_transforms = Compose(
    [
        ScaleIntensity(),
        EnsureChannelFirst(),
        Resize((224, 224, 224)),
        RandRotate90(),
        CustomToOneChannel(),
    ]
)

val_transforms = Compose(
    [
        ScaleIntensity(),
        EnsureChannelFirst(),
        Resize((224, 224, 224)),
        CustomToOneChannel(),
    ]
)

check_ds = ImageDataset(
    image_files=image_paths, labels=labels, transform=train_transforms
)
check_loader = DataLoader(check_ds, batch_size=3, num_workers=2, pin_memory=pin_memory)

im, label = monai.utils.misc.first(check_loader)
print(type(im), im.shape, label, label.shape)

# create a training data loader
train_ds = ImageDataset(
    image_files=image_paths_train,
    labels=labels_train,
    transform=train_transforms,
)
train_loader = DataLoader(
    train_ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=pin_memory
)

# create a validation data loader
val_ds = ImageDataset(
    image_files=image_paths_valid,
    labels=labels_valid,
    transform=val_transforms,
)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, pin_memory=pin_memory)

# create a test data loader
test_ds = ImageDataset(
    image_files=image_paths_test,
    labels=labels_test,
    transform=val_transforms,
)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=4, pin_memory=pin_memory)

# start a typical PyTorch training
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
writer = SummaryWriter()
step = -1

for batch_data in tqdm(train_loader):
    step += 1
