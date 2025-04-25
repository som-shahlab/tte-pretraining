import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from monai.data import DataLoader, ImageDataset
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
)
from monai.apps import download_and_extract
from sklearn.linear_model import LogisticRegressionCV
from src.utils import CustomToOneChannel, run_analysis
from src.networks import DenseNet121
import os


def train_model():
    parser = argparse.ArgumentParser(
        description="3D Brain MRI PyTorch Training with Analysis"
    )
    parser.add_argument(
        "--root_dir", type=str, default="./data_temp", help="Root directory for data"
    )
    parser.add_argument(
        "--loadmodel_path",
        type=str,
        default="./data_temp/ckpt.pth",
        help="Path to model weights",
    )
    parser.add_argument(
        "--mirror_ixi_data",
        type=str,
        default="https://drive.google.com/file/d/1f5odq9smadgeJmDeyEy_UOjEtE_pkKc0/view?usp=sharing",
        help="Mirror link to IXI data if https://brain-development.org/ixi-dataset/ no longer available",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")

    args = parser.parse_args()

    pin_memory = torch.cuda.is_available()
    device = torch.device("cuda:0")
    root_dir = args.root_dir

    # IXI dataset as a demo, downloadable from https://brain-development.org/ixi-dataset/
    # the path of ixi IXI-T1 dataset
    images = [
        os.sep.join([root_dir, "ixi", "IXI314-IOP-0889-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI249-Guys-1072-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI609-HH-2600-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI173-HH-1590-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI020-Guys-0700-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI342-Guys-0909-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI134-Guys-0780-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI577-HH-2661-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI066-Guys-0731-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI130-HH-1528-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI607-Guys-1097-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI175-HH-1570-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI385-HH-2078-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI344-Guys-0905-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI409-Guys-0960-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI584-Guys-1129-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI253-HH-1694-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI092-HH-1436-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI574-IOP-1156-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI585-Guys-1130-T1.nii.gz"]),
    ]

    # If Dataset not longer downloadable from the original source (https://brain-development.org/ixi-dataset/), use mirror link
    if not os.path.isfile(images[0]):
        resource = args.mirror_ixi_data
        md5 = "34901a0593b41dd19c1a1f746eac2d58"
        dataset_dir = os.path.join(root_dir, "ixi")
        tarfile_name = f"{dataset_dir}.tar"
        download_and_extract(resource, tarfile_name, dataset_dir, md5)

    labels = np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    labels = torch.nn.functional.one_hot(torch.as_tensor(labels)).float()

    train_transforms = Compose(
        [
            ScaleIntensity(),
            EnsureChannelFirst(),
            Resize((224, 224, 200)),
            RandRotate90(),
            CustomToOneChannel(),
        ]
    )
    val_transforms = Compose(
        [
            ScaleIntensity(),
            EnsureChannelFirst(),
            Resize((224, 224, 200)),
            CustomToOneChannel(),
        ]
    )

    train_ds = ImageDataset(
        image_files=images[:10], labels=labels[:10], transform=train_transforms
    )
    val_ds = ImageDataset(
        image_files=images[10:], labels=labels[10:], transform=val_transforms
    )

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=pin_memory
    )
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2, pin_memory=pin_memory)

    model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    writer = SummaryWriter()

    # Training loop
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    max_epochs = args.epochs

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        step = 0
        features_train = []
        labels_train = []

        # Training loop
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs, features = model(inputs, return_features=True)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            writer.add_scalar(
                "train_loss", loss.item(), epoch * len(train_loader) + step
            )

            # Collect features and labels for training data analysis
            features_train.append(features.detach().cpu().numpy())
            labels_train.append(torch.argmax(labels, dim=1).cpu().numpy())

        features_train = np.concatenate(features_train, axis=0)
        labels_train = np.concatenate(labels_train, axis=0)

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # Train logistic regression on extracted features from the training data
        linear_model = LogisticRegressionCV(
            penalty="l2", solver="liblinear", cv=2, n_jobs=-1
        ).fit(features_train, labels_train)

        # Validation step
        if (epoch + 1) % 2 == 0:
            model.eval()
            num_correct = 0.0
            metric_count = 0
            features_val = []
            labels_val = []

            for val_data in val_loader:
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                labels_val.append(torch.argmax(val_labels, dim=1).cpu().numpy())
                with torch.no_grad():
                    val_outputs, features = model(val_images, return_features=True)
                    features_val.append(features.cpu().numpy())
                    value = torch.eq(
                        val_outputs.argmax(dim=1), val_labels.argmax(dim=1)
                    )
                    metric_count += len(value)
                    num_correct += value.sum().item()

            features_val = np.concatenate(features_val, axis=0)
            labels_val = np.concatenate(labels_val, axis=0)

            # Calculate validation accuracy
            metric = num_correct / metric_count
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1

            writer.add_scalar("val_accuracy", metric, epoch + 1)
            print(f"Current accuracy: {metric:.4f} Best accuracy: {best_metric:.4f}")

            # Run analysis using logistic regression
            y_train_proba = linear_model.predict_proba(features_train)[:, 1]
            y_val_proba = linear_model.predict_proba(features_val)[:, 1]
            run_analysis(
                "Logistic Regression",
                labels_train,
                y_train_proba,
                labels_val,
                y_val_proba,
                label_col="gender",
            )

    writer.close()


if __name__ == "__main__":
    train_model()
