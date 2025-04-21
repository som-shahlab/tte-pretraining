from networks import filter_swinunetr
import torch
from collections import OrderedDict
import numpy as np
from monai.apps import download_url
import os
# import pydicom
from torch.utils.data import Dataset
import tarfile
import pandas as pd
from datetime import datetime
from monai.data import DataLoader, ImageDataset
import tempfile
from monai.transforms import LoadImage, Transform
import datasets
import femr
import femr.splits
import pickle
import shutil
import os
import meds
import sklearn
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


seed = 42
import random
import torch
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# If using CUDA
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="The max_iter was reached which means the coef_ did not converge")
ConvergenceWarning('ignore')
from warnings import simplefilter
simplefilter("ignore", category=ConvergenceWarning)

import torchtuples as tt
from pycox.models import CoxPH
import numpy as np
import matplotlib.pyplot as plt
# from pycox.evaluation import EvalSurv
# from utils import EvalSurv


# Ignore ConvergenceWarning
warnings.filterwarnings("ignore", category = ConvergenceWarning)

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

from sklearn.model_selection import GridSearchCV



def load_pretrained_swinunetr(model, use_pretrained, pretrained_path):
    # use_pretrained = True

    # if use_pretrained:
    #     resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/ssl_pretrained_weights.pth"
    #     dst = "/share/pi/nigam/projects/zphuo/data/PE/Jose_monai_MRI/ssl_pretrained_weights.pth"
    #     download_url(resource, dst)
    #     pretrained_path = os.path.normpath(dst)

    # Load SwinUNETR backbone weights into SwinUNETR
    if use_pretrained is True:
        print("Loading Weights from the Path {}".format(pretrained_path))
        ssl_dict = torch.load(pretrained_path)
        ssl_weights = ssl_dict["model"]
        
        # print('ssl_weights.keys()=====', ssl_weights.keys(), )
        # print('model.state_dict().keys()====', model.state_dict().keys(), )

        # Generate new state dict so it can be loaded to MONAI SwinUNETR Model
        monai_loadable_state_dict = OrderedDict()
        model_prior_dict = model.state_dict()
        model_update_dict = model_prior_dict

        del ssl_weights["encoder.mask_token"]
        del ssl_weights["encoder.norm.weight"]
        del ssl_weights["encoder.norm.bias"]
        del ssl_weights["out.conv.conv.weight"]
        del ssl_weights["out.conv.conv.bias"]

        for key, value in ssl_weights.items():
            if key[:8] == "encoder.":
                if key[8:19] == "patch_embed":
                    new_key = "swin_unetr.swinViT." + key[8:]
                else:
                    new_key = "swin_unetr.swinViT." + key[8:18] + key[20:]
                monai_loadable_state_dict[new_key] = value
                
            elif key.startswith("encoder"):
                key = key.replace("encoder", "swin_unetr.encoder")
                monai_loadable_state_dict[key] = value
            elif key.startswith("decoder"):
                key = key.replace("decoder", "swin_unetr.decoder")
                monai_loadable_state_dict[key] = value
            elif '0.0.blocks' in key:
                key = key.replace("0.0.blocks", "0.blocks")
                monai_loadable_state_dict[key] = value
            else:
                monai_loadable_state_dict[key] = value

        model_update_dict.update(monai_loadable_state_dict)
        # model.load_state_dict(model_update_dict, strict=True)
        # model.load_state_dict(model_update_dict, strict=False)
        model_final_loaded_dict = model.state_dict()

        # Safeguard test to ensure that weights got loaded successfully
        layer_counter = 0
        for k, _v in model_final_loaded_dict.items():
            if k in model_prior_dict:
                layer_counter = layer_counter + 1

                old_wts = model_prior_dict[k]
                new_wts = model_final_loaded_dict[k]

                old_wts = old_wts.to("cpu").numpy()
                new_wts = new_wts.to("cpu").numpy()
                diff = np.mean(np.abs(old_wts, new_wts))
                print("Layer {}, the update difference is: {}".format(k, diff))
                if diff == 0.0:
                    print("Warning: No difference found for layer {}".format(k))
        print(
            "Total updated layers {} / {}".format(layer_counter, len(model_prior_dict))
        )
        print("Pretrained Weights Succesfully Loaded !")

    elif use_pretrained is False:
        print(
            "No weights were loaded, all weights being used are randomly initialized!"
        )

    return model


class DicomDataset(Dataset):
    def __init__(self, dicom_folder, transforms):
        self.dicom_folder = dicom_folder
        self.transforms = transforms
        self.dicom_files = [
            os.path.join(dicom_folder, f)
            for f in os.listdir(dicom_folder)
            if f.endswith(".dcm")
        ]

    def __len__(self):
        return len(self.dicom_files)

    def __getitem__(self, idx):
        dicom_path = self.dicom_files[idx]
        dicom_image = pydicom.dcmread(dicom_path).pixel_array
        dicom_image = self.transforms(dicom_image)
        return dicom_image


class TarImageDataset(ImageDataset):
    def __init__(self, tar_files, labels, transform=None, **kwargs):
        """
        Initializes the dataset with TAR files.
        tar_files: A list of paths to TAR files.
        labels: A list of labels corresponding to each image.
        transform: Transformations to apply to each image.
        """

        super().__init__(
            image_files=tar_files, labels=labels, transform=transform, **kwargs
        )
        self.loader = LoadImage(
            image_only=True
        )  # Use MONAI's LoadImage for loading images

        self.AddChannel = AddChannelTransform()

    def __getitem__(self, index):
        """
        Overrides the __getitem__ method to read images from a TAR file.
        """
        # TAR file path
        tar_file_path = self.image_files[index]
        # Label for the current image
        label = self.labels[index] if self.labels is not None else None

        dicom_images = self.extract_dicoms_from_tar(tar_file_path)

        # # Add the transform to your pipeline if your data lacks a channel dimension
        # print(dicom_images.shape, "before")
        # dicom_images = self.AddChannel(dicom_images)
        # print(dicom_images.shape, "after")

        dicom_images_path = self.extract_dicoms_from_tar_return_path(
            tar_file_path, os.path.dirname(tar_file_path)
        )
        dicom_images = self.loader(dicom_images_path)

        # Apply transformations
        if self.transform is not None:
            dicom_images = self.transform(dicom_images)
        return tuple([dicom_images, self.labels[index]])

    def extract_dicoms_from_tar(self, tar_path):
        with tarfile.open(tar_path) as tar:
            dicom_images = []
            for member in tar.getmembers():
                if member.name.endswith(".dcm"):
                    f = tar.extractfile(member)
                    if f is not None:
                        dicom = pydicom.dcmread(f)
                        # Convert pixel array to float32
                        image = dicom.pixel_array.astype(np.float32)
                        # image /= np.max(image)  # Simple normalization to [0, 1]
                        dicom_images.append(image)
            # Stack slices to create a 3D volume
            if dicom_images:
                volume = np.stack(dicom_images, axis=0)
                # volume = np.expand_dims(volume, axis=0)
                return volume
            else:
                return np.zeros(
                    (1, 512, 512), dtype=np.float32
                )  # Adjust dimensions as needed

    def extract_dicoms_from_tar_return_path(self, tar_path, extract_path):
        # check extract_path exisits
        if not os.path.exists(extract_path):
            os.makedirs(extract_path, exist_ok=True)

        extracted_image_paths = []
        with tarfile.open(tar_path, "r:*") as tar:
            tar.extractall(path=extract_path)
            extracted_image_paths.extend(
                [
                    os.path.join(extract_path, member.name)
                    for member in tar.getmembers()
                    if member.isdir() and member.name != "."
                ]
            )

        return extracted_image_paths


'''       
class TarImageDataset(ImageDataset):
    def __init__(self, tar_files, labels, transform=None, image_only=True, **kwargs):
        """
        tar_files: List of paths to the TAR files containing the images.
        labels: The labels corresponding to each image in the TAR files.
        transform: Transformations to be applied to the images.
        image_only: Whether to return only the images without the metadata.
        """
        self.temp_dir = tempfile.TemporaryDirectory()  # Temporary directory for extracted files
        extracted_image_paths = self.extract_tar_files(tar_files, self.temp_dir.name)
        
        # Now, call the superclass initializer with the paths to the extracted images
        super().__init__(image_files=extracted_image_paths, labels=labels, transform=transform, image_only=image_only, **kwargs)

    def extract_tar_files(self, tar_files, extract_path):
        """
        Extracts images from the TAR files and returns the paths to the extracted images.
        """
        extracted_image_paths = []
        for tar_file in tar_files:
            with tarfile.open(tar_file, "r:*") as tar:
                tar.extractall(path=extract_path)
                extracted_image_paths.extend([os.path.join(extract_path, member.name) for member in tar.getmembers() if member.isfile()])
        return extracted_image_paths

    def __del__(self):
        """
        Cleanup the temporary directory upon deletion of the dataset object.
        """
        self.temp_dir.cleanup()
'''


class AddChannelTransform(Transform):
    def __call__(self, img):
        if len(img.shape) == 3:  # for 3D volumes
            return img[None, :]  # add a channel dimension
        elif len(img.shape) == 2:  # for 2D images
            return img[None, :, :]  # add a channel dimension
        else:
            return img  # return as is if the dimension is unexpected

def squeeze_1_channel(images):
    if images.shape[1] != 1:
        images = images.mean(dim=1, keepdim=True)
    return images
        
class CustomToOneChannel(Transform):
    def __call__(self, img):
        # Implement your logic here
        # Example: averaging the channels
        if img.shape[0] == 1:
            return img
        else:
            return img.mean(dim=0, keepdim=True)
        
class CustomToOneChanneld(Transform):

    def __init__(self, keys):
 
        self.keys = keys

    def __call__(self, data):
        
        # Ensure the input is a dictionary
        if not isinstance(data, dict):
            raise TypeError("Input data should be a dictionary")

        # Iterate over all specified keys and apply the transformation
        for key in self.keys:
            img = data[key]
            if img.shape[0] == 1:
                data[key] = img
            else:
                data[key] = img.mean(dim=0, keepdim=True)
        
        return data

def MOTORloss(outputs, targets):
    # Define the loss function
    loss = nn.CrossEntropyLoss()
    # Compute the loss value
    loss_value = loss(outputs, targets)
    return loss_value


import xgboost as xgb
import sklearn.linear_model
import sklearn.metrics
import sklearn.preprocessing

def run_analysis(title: str, y_train, y_train_proba, y_test=None, y_test_proba=None, label_col = 'finetune_label', confidence_interval=True):
    if y_test is None or y_test_proba is None:
        print(f"---- {title} {label_col} ----")
        print("Test:")
        auroc = print_metrics(y_train, y_train_proba)
        return auroc
    print(f"---- {title} {label_col} ----")
    print("Train:")
    auroc_train = print_metrics(y_train, y_train_proba)
    print("Test:")
    auroc = print_metrics(y_test, y_test_proba)
    if confidence_interval:
        auroc_train_ci, auroc_ci = bootstrap_auc(y_train, y_train_proba), bootstrap_auc(y_test, y_test_proba)
        print("Train CI:", auroc_train_ci, "Test CI:", auroc_ci)
        return auroc_train, auroc, auroc_train_ci, auroc_ci
    else:
        return auroc_train, auroc

def bootstrap_auc(y_true, y_pred, n_bootstraps=1000, rng_seed=42):
    bootstrapped_scores = []
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = sklearn.metrics.roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
        # print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

        ci=np.quantile(bootstrapped_scores, (0.05, 0.95))
    return ci

def bootstrap_tdcs(times, is_censor, hazards, n_bootstraps=1000, rng_seed=42):
    bootstrapped_scores = []
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(times), len(times))
        

        score = test_c_statistic(times[indices], is_censor[indices], hazards[indices])
        bootstrapped_scores.append(score)

        ci=np.quantile(bootstrapped_scores, (0.05, 0.95))
    return ci

def bootstrap_ibs(model, X_test, y_test, n_bootstraps=1000, rng_seed=42):
    bootstrapped_scores = []
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(X_test), len(X_test))
        
        surv = model.predict_surv_df(X_test[indices])
        ev = EvalSurv(surv, y_test[0][indices], y_test[1][indices], censor_surv='km')
        time_grid = np.linspace(y_test[0].min(), y_test[0].max(), 100)
        score = ev.integrated_brier_score(time_grid)
        bootstrapped_scores.append(score)
        
        ci=np.quantile(bootstrapped_scores, (0.05, 0.95))
    return ci

def print_metrics(y_true, y_proba):
    y_pred = y_proba > 0.5

    auroc = sklearn.metrics.roc_auc_score(y_true, y_proba)
    aps = sklearn.metrics.average_precision_score(y_true, y_proba)
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    f1 = sklearn.metrics.f1_score(y_true, y_pred)
    auprc = sklearn.metrics.precision_recall_curve(y_true, y_proba)
    print("\tAUROC:", auroc)
    print("\tAPS:", aps)
    print("\tAccuracy:", accuracy)
    print("\tF1 Score:", f1)
    return auroc


def set_up_motor_task(TARGET_DIR, from_pretrained_tokenizer, month_date_hour, num_proc, label_csv, ontology_path, inference, vocab_size, START_TIME, parquet_folder, final_layer_size, num_tasks, test_subset):
    ###### set up MOTOR task ########
    ################################
    


    # os.environ["HF_DATASETS_CACHE"] = '/share/pi/nigam/zphuo/cache_dir'

    # TARGET_DIR = '/share/pi/nigam/projects/zphuo/repos/PE_3D_multimodal/training/trash/tutorial_6_INSEPCT'

    from_pretrained = from_pretrained_tokenizer
    # num_proc = 20
    
    # if not from_pretrained:
    #     if os.path.exists(TARGET_DIR):
    #         shutil.rmtree(TARGET_DIR)

    if month_date_hour is None:
        month_date_hour = datetime.now().strftime("%m%d%H")
    # if path not exists, create the folder
    if not os.path.exists(TARGET_DIR):
        os.mkdir(TARGET_DIR)
    if not os.path.exists(os.path.join(TARGET_DIR, f'motor_model_{month_date_hour}')):
        os.mkdir(os.path.join(TARGET_DIR, f'motor_model_{month_date_hour}'))

    
    # test subset or not
    if not test_subset:
        parquet_folder_full = os.path.join(parquet_folder, 'data', '*')
        label_csv_full = label_csv.replace('_subset', '')
    else:
        parquet_folder_full = None
        label_csv_full = None

    # dataset = datasets.Dataset.from_parquet('input/meds/data/*')
    if 'subset' in label_csv:
        #parquet_folder = '/share/pi/nigam/projects/zphuo/data/PE/inspect/timelines_smallfiles_meds/data_subset/*'
        parquet_folder = os.path.join(parquet_folder, 'data_subset', '*')
        
    else:
        #parquet_folder = '/share/pi/nigam/projects/zphuo/data/PE/inspect/timelines_smallfiles_meds/data/*'
        parquet_folder = os.path.join(parquet_folder, 'data', '*')
    dataset = datasets.Dataset.from_parquet(parquet_folder)
    if not test_subset:
        dataset_full = datasets.Dataset.from_parquet(parquet_folder_full)


    print('indexing patients...')
    index = femr.index.PatientIndex(dataset, num_proc=num_proc)
    if not test_subset:
        index_full = femr.index.PatientIndex(dataset_full, num_proc=num_proc)
    print('time used indexing patients:', datetime.now() - START_TIME)
    
    if 'subset' not in label_csv:
        inspect_split_csv = os.path.join(TARGET_DIR, f'motor_model_{month_date_hour}', "main_split.csv")
    else:
        inspect_split_csv = os.path.join(TARGET_DIR, f'motor_model_{month_date_hour}', "main_split_subset.csv")
        
    if not test_subset:
        inspect_split_csv_full = inspect_split_csv.replace('_subset', '')
    
    
    if not from_pretrained:
        # main_split = femr.splits.generate_hash_split(index.get_patient_ids(), 97, frac_test=0.15)
        # Note that we want to save this to the target directory since this is important information
        # main_split.save_to_csv(os.path.join(TARGET_DIR, "motor_model", "main_split.csv"))
        # train_split = femr.splits.generate_hash_split(main_split.train_patient_ids, 87, frac_test=0)

        # print(train_split.train_patient_ids)
        # print(train_split.test_patient_ids)

        label_csv_subset = label_csv
        label_df = pd.read_csv(label_csv_subset)
        label_df = label_df[['patient_id', 'split', ]]
        label_df = label_df.rename(columns={'split': 'split_name'})
        label_df.to_csv(inspect_split_csv, index=False)
        
        if not test_subset:
            label_csv_subset = label_csv.replace('_subset', '')
            label_df = pd.read_csv(label_csv_subset)
            label_df = label_df[['patient_id', 'split', ]]
            label_df = label_df.rename(columns={'split': 'split_name'})
            inspect_split_csv_full = inspect_split_csv.replace('_subset', '')
            label_df.to_csv(inspect_split_csv_full, index=False)
            
        
    main_split = femr.splits.PatientSplit.load_from_csv(inspect_split_csv)
    if not test_subset:
        main_split_full = femr.splits.PatientSplit.load_from_csv(inspect_split_csv_full)

    main_dataset = main_split.split_dataset(dataset, index)
    if not test_subset:
        main_dataset_full = main_split_full.split_dataset(dataset_full, index_full)
    train_dataset = main_dataset['train']
    
    if test_subset:
        valid_dataset = main_dataset['valid']
        test_dataset = main_dataset['test']
    else:
        valid_dataset = main_dataset_full['valid']
        test_dataset = main_dataset_full['test']
    
    '''delete
    train_split = femr.splits.generate_hash_split(main_split.train_patient_ids, 87, frac_test=0)
    valid_split = femr.splits.generate_hash_split(main_split.valid_patient_ids, 87, frac_test=0)
    test_split = femr.splits.generate_hash_split(main_split.test_patient_ids, 87, frac_test=0)
    
    print('splitting dataset to train...')
    train_dataset = train_split.split_dataset(main_dataset['train'], femr.index.PatientIndex(main_dataset['train'], num_proc=num_proc))
    print('time used splitting train dataset:', datetime.now() - START_TIME)
    
    print('splitting dataset to valid...')
    valid_dataset = valid_split.split_dataset(main_dataset['valid'], femr.index.PatientIndex(main_dataset['valid'], num_proc=num_proc))
    print('time used splitting valid dataset:', datetime.now() - START_TIME)
    
    print('splitting dataset to test...')
    test_dataset = test_split.split_dataset(main_dataset['test'], femr.index.PatientIndex(main_dataset['test'], num_proc=num_proc))
    print('time used splitting test dataset:', datetime.now() - START_TIME)
    '''
    
    # First, we need to train a tokenizer
    # Note, we need to use a hierarchical tokenizer for MOTOR


    with open(ontology_path, 'rb') as f:
        ontology = pickle.load(f)
    if (not from_pretrained) and (not inference):
        print("Training tokenizer...")
        tokenizer = femr.models.tokenizer.train_tokenizer(
            main_dataset['train'], vocab_size=vocab_size, is_hierarchical=True, num_proc=num_proc, ontology=ontology)

        # Save the tokenizer to the same directory as the model
        tokenizer.save_pretrained(os.path.join(TARGET_DIR, f'motor_model_{month_date_hour}'))

    else:
        # load pretrained tokenizer
        tokenizer = femr.models.tokenizer.FEMRTokenizer.from_pretrained(os.path.join(TARGET_DIR, f'motor_model_{month_date_hour}'), ontology=ontology)
    

    if 'subset' in parquet_folder:
        num_tasks = 39
    else:
        num_tasks = num_tasks
        # num_tasks = 8192
        # num_tasks = 128

    # Second, we need to prefit the MOTOR model. This is necessary because piecewise exponential models are unstable without an initial fit

    time_used = datetime.now() - START_TIME
    print(f"Time used tokenzier: {time_used}")
    
    print("Prefitting MOTOR task...")
    if not from_pretrained:
    # if True:
        motor_task = femr.models.tasks.MOTORTask.fit_pretraining_task_info(
            main_dataset['train'], tokenizer, num_tasks=num_tasks, num_bins=8, final_layer_size=final_layer_size, num_proc=num_proc)
        with open(os.path.join(TARGET_DIR, f'motor_model_{month_date_hour}', "motor_task.pkl"), 'wb') as f:
            pickle.dump(motor_task, f)
    else:
        with open(os.path.join(TARGET_DIR, f'motor_model_{month_date_hour}', "motor_task.pkl"), 'rb') as f:
            motor_task = pickle.load(f)

    time_used = datetime.now() - START_TIME
    print(f"Time used motor task: {time_used}")
    
    # Third, we need to create batches. 

    processor = femr.models.processor.FEMRBatchProcessor(tokenizer, motor_task)

    index_train = femr.index.PatientIndex(train_dataset, num_proc=num_proc)
    index_valid = femr.index.PatientIndex(valid_dataset, num_proc=num_proc)
    index_test = femr.index.PatientIndex(test_dataset, num_proc=num_proc)
    # We can do this one patient at a time

    time_used = datetime.now() - START_TIME
    print(f"Time used index: {time_used}")
    
    # transformer_config = femr.models.transformer.FEMRTransformerConfig(
    # vocab_size=tokenizer.vocab_size, 
    # is_hierarchical=tokenizer.is_hierarchical, 
    # n_layers=2,
    # hidden_size=764, 
    # intermediate_size=64*2,
    # n_heads=8,
    # )
    
    # config = femr.models.transformer.FEMRModelConfig.from_transformer_task_configs(transformer_config, motor_task.get_task_config())
    
    ###### set up MOTOR task ########
    ################################
    
    return motor_task, tokenizer, train_dataset, valid_dataset, test_dataset, processor, index_train, index_valid, index_test, num_tasks, ontology


def make_as_tensor(motor_batch):
    for key in motor_batch:
        if type(motor_batch[key]) == dict:
            motor_batch[key] = {k: torch.from_numpy(v)for k, v in motor_batch[key].items()}
        elif type(motor_batch[key]) == np.ndarray:
            motor_batch[key] = torch.from_numpy(motor_batch[key])
    return motor_batch

def remove_prefix_from_state_dict(state_dict, prefix='module.'):
    """Remove prefix from the state_dict keys."""
    return {k.replace(prefix, ''): v for k, v in state_dict.items() if k.startswith(prefix)}


def convert_3d_to_2d_weights(ckpt_3d, model_2d):
    new_state_dict = model_2d.state_dict()
    
    for key in ckpt_3d:
        if key in new_state_dict:  # Ensure the key exists in the model's state_dict
            if 'conv' in key or 'downsample.0' in key:
                # Converting convolution weights
                weight_3d = ckpt_3d[key]
                if len(weight_3d.shape) == 5:  # Only convert if it's a 3D conv weight
                    weight_2d = weight_3d.sum(dim=2)  # Sum across the depth dimension
                    new_state_dict[key] = weight_2d
                else:
                    new_state_dict[key] = weight_3d
            elif 'bn' in key or 'downsample.1' in key or 'norm' in key:
                # BatchNorm and normalization layers
                new_state_dict[key] = ckpt_3d[key]
            elif 'fc' in key:
                # Fully connected layers
                new_state_dict[key] = ckpt_3d[key]
            else:
                print(f"Skipping layer {key} with shape {ckpt_3d[key].shape}")
        else:
            print(f"Key {key} not found in model's state_dict")

    return new_state_dict


def convert_3d_to_2d_weights_densenet(ckpt_3d, model_2d):
    new_state_dict = model_2d.state_dict()
    
    for key in ckpt_3d:
        if key in new_state_dict:  # Ensure the key exists in the model's state_dict
            if 'conv' in key or 'downsample.0' in key:
                # Converting convolution weights
                weight_3d = ckpt_3d[key]
                if len(weight_3d.shape) == 5:  # Only convert if it's a 3D conv weight
                    if 'conv0.weight' in key:  # Initial convolution layer
                        weight_2d = weight_3d.squeeze(2)  # Remove the depth dimension
                        if weight_2d.shape[1] != new_state_dict[key].shape[1]:  # Handle channel mismatch
                            if weight_2d.shape[1] == 3 and new_state_dict[key].shape[1] == 1:
                                # If converting from RGB to grayscale, average the weights across the color channels
                                weight_2d = weight_2d.mean(dim=1, keepdim=True)
                            elif weight_2d.shape[1] == 1 and new_state_dict[key].shape[1] == 3:
                                # If converting from grayscale to RGB, duplicate the single channel
                                weight_2d = weight_2d.repeat(1, 3, 1, 1)
                        new_state_dict[key] = weight_2d
                    else:
                        weight_2d = weight_3d.squeeze(2)  # Remove the depth dimension
                    new_state_dict[key] = weight_2d
                else:
                    new_state_dict[key] = weight_3d
            elif 'bn' in key or 'downsample.1' in key or 'norm' in key:
                # BatchNorm and normalization layers
                new_state_dict[key] = ckpt_3d[key]
            elif 'fc' in key:
                # Fully connected layers
                new_state_dict[key] = ckpt_3d[key]
            else:
                print(f"Skipping layer {key} with shape {ckpt_3d[key].shape}")
        else:
            # Handle deeper nested keys by removing one level and matching
            modified_key = key.replace('layers.layers.', 'layers.')
            if modified_key in new_state_dict:
                if 'conv' in modified_key or 'downsample.0' in modified_key:
                    weight_3d = ckpt_3d[key]
                    if len(weight_3d.shape) == 5:
                        if 'conv0.weight' in modified_key:
                            weight_2d = weight_3d.squeeze(2)
                            if weight_2d.shape[1] != new_state_dict[modified_key].shape[1]:  # Handle channel mismatch
                                if weight_2d.shape[1] == 3 and new_state_dict[modified_key].shape[1] == 1:
                                    weight_2d = weight_2d.mean(dim=1, keepdim=True)
                                elif weight_2d.shape[1] == 1 and new_state_dict[modified_key].shape[1] == 3:
                                    weight_2d = weight_2d.repeat(1, 3, 1, 1)
                            new_state_dict[modified_key] = weight_2d
                        else:
                            weight_2d = weight_3d.sum(dim=2)
                        new_state_dict[modified_key] = weight_2d
                    else:
                        new_state_dict[modified_key] = weight_3d
                elif 'bn' in modified_key or 'downsample.1' in modified_key or 'norm' in modified_key:
                    new_state_dict[modified_key] = ckpt_3d[key]
                elif 'fc' in modified_key:
                    new_state_dict[modified_key] = ckpt_3d[key]
            else:
                pass
                # print(f"Key {key} not found in model's state_dict, even after modification")

    return new_state_dict

def load_different_model(model, pretrained_dict, ddp=False):
    # print(pretrained_dict.keys())
    # if ddp:
    if 'module.' in list(pretrained_dict.keys())[0]:
        pretrained_dict = remove_prefix_from_state_dict(pretrained_dict)
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model.state_dict()}
    pretrain_unique_dict = {k: v for k, v in pretrained_dict.items() if k not in model.state_dict()}
    model_unique_dict = {k: v for k, v in model.state_dict().items() if k not in pretrained_dict}
    if len(pretrain_unique_dict) > 0 or len(model_unique_dict) > 0:
        print('weights not loaded:', list(pretrain_unique_dict.keys())[:20])
        print('weights initialized:', list(model_unique_dict.keys())[:20])
    else:
        print('all weights loaded')
    print('weights loaded:', list(filtered_dict.keys())[:20], len(filtered_dict))
    try:
        model.load_state_dict(filtered_dict, strict=False)
    except:
        # classification layer no need to load
        filtered_dict = {k: v for k, v in pretrained_dict.items() if 'classification' not in k}
    return model


def load_different_model_2D(model, pretrained_dict, ddp=False):
    # print(pretrained_dict.keys())
    # if ddp:
    if 'module.' in list(pretrained_dict.keys())[0]:
        pretrained_dict = remove_prefix_from_state_dict(pretrained_dict)
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model.state_dict()}
    pretrain_unique_dict = {k: v for k, v in pretrained_dict.items() if k not in model.state_dict()}
    model_unique_dict = {k: v for k, v in model.state_dict().items() if k not in pretrained_dict}
    if len(pretrain_unique_dict) > 0 or len(model_unique_dict) > 0:
        print('weights not loaded:', list(pretrain_unique_dict.keys())[:20])
        print('weights initialized:', list(model_unique_dict.keys())[:20])
    else:
        print('all weights loaded')
    print('weights loaded:', list(filtered_dict.keys())[:20], len(filtered_dict))
    model.load_state_dict(filtered_dict, strict=False)
    return model

def load_different_model_i3densenet(model, pretrained_dict, ddp=False):
    # print(pretrained_dict.keys())
    # if ddp:
    if 'module.' in list(pretrained_dict.keys())[0]:
        pretrained_dict = remove_prefix_from_state_dict(pretrained_dict)
    pretrained_dict_new = {}
    
    # handle '.denselayer1.layers' 
    for k, v in pretrained_dict.items():
        if 'denselayer' in k:
            k_ls = k.split('.')
            for i in range(len(k_ls)):
                if k_ls[i].startswith('denselayer'):
                    k_ls[i] += '.layers'
            k_new = '.'.join(k_ls)
            pretrained_dict_new[k_new] = v
        else:
            pretrained_dict_new[k] = v
    pretrained_dict = pretrained_dict_new
            
    # handle conv0, RGB to grey scale
    rgb_weights = pretrained_dict['features.conv0.weight']
    gray_weights = rgb_weights.mean(dim=1, keepdim=True)
    gray_weights = gray_weights.repeat(1, 1, 7, 1, 1)
    pretrained_dict['features.conv0.weight'] = gray_weights
    
    
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model.state_dict()}
    pretrain_unique_dict = {k: v for k, v in pretrained_dict.items() if k not in model.state_dict()}
    model_unique_dict = {k: v for k, v in model.state_dict().items() if k not in pretrained_dict}
    if len(pretrain_unique_dict) > 0 or len(model_unique_dict) > 0:
        print('weights not loaded:', list(pretrain_unique_dict.keys())[:20])
        print('weights initialized:', list(model_unique_dict.keys())[:20])
    else:
        print('all weights loaded')
    print('weights loaded:', list(filtered_dict.keys())[:20], len(filtered_dict))
    model.load_state_dict(filtered_dict, strict=False)
    return model


def load_different_model_i3resnet(model, pretrained_dict, ddp=False):
    # print(pretrained_dict.keys())
    # if ddp:
    if 'module.' in list(pretrained_dict.keys())[0]:
        pretrained_dict = remove_prefix_from_state_dict(pretrained_dict)
    pretrained_dict_new = {}
    
    conv1_weight = pretrained_dict['conv1.weight']
    if conv1_weight.shape[1] == 3 and model.conv1.weight.shape[1] == 1:
        print(f'Adjusting conv1 weights from shape {conv1_weight.shape} to {model.conv1.weight.shape}')
        # Average the 3 channels to convert from RGB to single-channel
        conv1_weight = conv1_weight.mean(dim=1, keepdim=True)
    
    if conv1_weight.shape[2] == 3 and model.conv1.weight.shape[2] == 7:
        print(f'Resizing conv1 kernel from 3x7x7 to 7x7x7')
        # Resize kernel using interpolation
        conv1_weight = F.interpolate(conv1_weight, size=(7, 7, 7), mode='trilinear', align_corners=False)
    
    # remove conv1.weight from pretrained_dict
    del pretrained_dict['conv1.weight']
    
    # handle fc layer[1000, 2048], [400, 2048]
    # for key, value in pretrained_dict.items():
    #     if "fc." in key and value.shape[0] != model.state_dict()[key].size():
    if 'fc.weight' in pretrained_dict:
        del pretrained_dict['fc.weight']
    if 'fc.bias' in pretrained_dict:
        del pretrained_dict['fc.bias']
    
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model.state_dict()}
    pretrain_unique_dict = {k: v for k, v in pretrained_dict.items() if k not in model.state_dict()}
    model_unique_dict = {k: v for k, v in model.state_dict().items() if k not in pretrained_dict}
    if len(pretrain_unique_dict) > 0 or len(model_unique_dict) > 0:
        print('weights not loaded:', list(pretrain_unique_dict.keys())[:20])
        print('weights initialized:', list(model_unique_dict.keys())[:20])
    else:
        print('all weights loaded')
    print('weights loaded:', list(filtered_dict.keys())[:20], len(filtered_dict))
    model.load_state_dict(filtered_dict, strict=False)
    return model


def get_final_batch(image_path, nii_folder, motor_task, dataset_split, index_split, ontology):
    patient_id_ls = []
    ct_time_ls = []
    for img_path in image_path:
        patient_id = int(img_path.replace(nii_folder+ "/", '').split('_')[0])
        patient_id_ls.append(patient_id)
        ct_time = ' '.join(img_path.replace(nii_folder+ "/", '').replace('.nii.gz', '').split('_')[1:])
        ct_time_ls.append(datetime.strptime(ct_time, '%Y-%m-%d %H %M %S'))


    # Handle multiple patients per batch. delete single one when done:
    # patient_id = int(image_paths_train[step].replace(nii_folder+ "/", '').split('_')[0])
    # ct_time = ' '.join(image_paths_train[step].replace(nii_folder+ "/", '').replace('.nii.gz', '').split('_')[1:])
    # ct_time = datetime.strptime(ct_time, '%Y-%m-%d %H %M %S')
    # last_time = ct_time


    final_batch_ls = []
    for patient_id, ct_time in zip(patient_id_ls, ct_time_ls):
        motor_task.start_batch()
        # print(patient_id, ct_time)
        patient: meds.Patient = dataset_split[index_split.get_index(patient_id)]

        motor_task.start_patient(patient, ontology)

        motor_task.add_event(ct_time, patient["events"][-1]['time'], None)
        
        motor_batch = motor_task.get_batch_data()
        # just to make as torch tensor
        motor_batch = make_as_tensor(motor_batch)
        # for key in motor_batch:
        #     if type(motor_batch[key]) == dict:
        #         motor_batch[key] = {k: torch.from_numpy(v)for k, v in motor_batch[key].items()}
        #     elif type(motor_batch[key]) == np.ndarray:
        #         motor_batch[key] = torch.from_numpy(motor_batch[key])

        one_final_batch = motor_task.cleanup(motor_batch)
        final_batch_ls.append(one_final_batch)

    final_batch = final_batch_ls[0] 
    for i in range(1, len(final_batch_ls), 1):
        one_final_batch = final_batch_ls[i]
        final_batch['is_event'] = torch.cat([final_batch['is_event'], one_final_batch['is_event']], dim=0)
        final_batch['log_time'] = torch.cat([final_batch['log_time'], one_final_batch['log_time']], dim=0)
        final_batch['time'] = torch.cat([final_batch['time'], one_final_batch['time']], dim=0)
        
        
    return final_batch, motor_batch

# Function to convert the string representation to a list of integers
def convert_to_list(item):
    # If the item is a numpy array containing lists, convert it to a list first
    if isinstance(item, np.ndarray):
        item = item.tolist()
    
    # If the item is still a nested list, extract the first element
    if isinstance(item, list) and len(item) == 1:
        item = item[0]

    # If the item is a list or numpy.int64, return it as is
    if isinstance(item, list):
        return item
    
    if isinstance(item, np.int64) or isinstance(item, int) or isinstance(item, np.float64) or isinstance(item, float):
        return [item]
    
    # If the item is a string, process it to convert to a list of integers
    if isinstance(item, str):
        s = item.strip("[]").split()
        value_mapping = {'0.0': 0, '1.0': 1, '0': 0, '1': 1, 'Censored': 'Censored', '-1.0': 1, 'True': 1, 'False': 0, 'nan': 0}
        return [value_mapping[x.strip("'")] for x in s]
        # return [int(x.strip("'")) for x in s]

    # Raise an error if the item is of an unsupported type
    raise ValueError(f"Unsupported type: {type(item)}")



def logistic_regression(column_ls, X_train, y_train, X_val, y_val, model_save_path, model_choice, epoch, month_date, metric_values, run, writer, tune_linearprobe, confidence_interval=False, prop_train=None, prop_test=None):
    
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    auroc_train_dict = {}
    auroc_val_dict = {}
    
    y_train_ = y_train
    y_val_ = y_val
    
    # y_train is a list
    if type(y_train) == list:
        linear_model = sklearn.linear_model.LogisticRegressionCV(penalty="l2", solver="liblinear").fit(X_train, y_train)
        
        y_train_proba = linear_model.predict_proba(X_train)[::, 1]
        y_val_proba = linear_model.predict_proba(X_val)[::, 1]
        if confidence_interval:
            auroc_train, auroc_val, auroc_train_ci, auroc_val_ci = run_analysis("Logistic Regression", y_train, y_train_proba, y_val, y_val_proba, confidence_interval=True)
            auroc_train_dict['auroc_train_finetune_label'] = auroc_train
            auroc_val_dict['auroc_val_finetune_label'] = auroc_val
            auroc_train_dict['auroc_train_ci_finetune_label'] = auroc_train_ci
            auroc_val_dict['auroc_val_ci_finetune_label'] = auroc_val_ci
        else:
            auroc_train, auroc_val = run_analysis("Logistic Regression", y_train, y_train_proba, y_val, y_val_proba, confidence_interval=False)
            auroc_train_dict['auroc_train_finetune_label'] = auroc_train
            auroc_val_dict['auroc_val_finetune_label'] = auroc_val
        # run.track(auroc_train, name='auroc_train', step=epoch, context={'subset': 'train'})
        # writer.add_scalar("val_accuracy", auroc_val, epoch)
        # run.track(auroc_val, name='auroc_val', step=epoch, context={'subset': 'valid'})
        
        linear_model_save_path = os.path.join(model_save_path, f"linear_model_{epoch}epoch_{model_choice}_{month_date}.pkl")
        with open(linear_model_save_path,'wb') as f:
            pickle.dump(linear_model, f)
    
        metric_values.append(auroc_val)
    
    # y_train is an array
    else:
        auroc_val_list = []
        
        for idx, label_col in enumerate(column_ls):
            if y_train_.ndim == 2 and y_train_.shape[1] == 1:
                y_train = y_train_.reshape(-1)
            if y_val_.ndim == 2 and y_val_.shape[1] == 1:
                y_val = y_val_.reshape(-1)
                
            if X_train.ndim == 3:
                X_train = X_train.reshape(-1, X_train.shape[2])
            if X_val.ndim == 3:
                X_val = X_val.reshape(-1, X_val.shape[2])
            
            if y_train.ndim == 1 or y_train.shape[1] == 1:
                nested_list = [convert_to_list(item) for item in y_train]
                y_train = np.array(nested_list)
            if y_val.ndim == 1 or y_val.shape[1] == 1:
                nested_list = [convert_to_list(item) for item in y_val]
                y_val = np.array(nested_list)
            # import pdb; pdb.set_trace()
            y_train_list = y_train[:, idx]
            idx_uncensored = np.where((y_train_list != 'Censored') & (y_train_list != -1))[0]
            
            X_train_list = X_train[idx_uncensored]
            y_train_list = y_train_list[idx_uncensored]
            
            y_train_list = y_train_list.astype(int)
            print(f'num of censored for {label_col} in train', len(y_train.tolist())-len(y_train_list), 'out of', len(y_train.tolist()), 'unique values and counts:', np.unique(y_train_list, return_counts=True))
            
            y_val_task = y_val[:, idx]
            idx_uncensored = np.where((y_val_task != 'Censored') & (y_val_task != -1))[0]
            X_val_list = X_val[idx_uncensored]
            y_val_task = y_val_task[idx_uncensored]
            y_val_task = y_val_task.astype(int)
            print(f'num of censored for {label_col} in val', len(y_val.tolist())-len(y_val_task), 'out of', len(y_val.tolist()), 'unique values and counts:', np.unique(y_val_task, return_counts=True))
            
            if prop_train:
                X_train_list = X_train_list[prop_train]
                y_train_list = y_train_list[prop_train]
            if prop_test:
                X_val_list = X_val_list[prop_test]
                y_val_task = y_val_task[prop_test]
            
            if tune_linearprobe:
                linear_model = sklearn.linear_model.LogisticRegressionCV(penalty="l2", solver="liblinear", cv=2, n_jobs=-1).fit(X_train_list, y_train_list)
                
                tree_model = xgb.XGBClassifier()
                tree_model = tune_hyperparameter_tree(tree_model, X_train_list, y_train_list)
            else:
                # linear_model = sklearn.linear_model.LogisticRegressionCV(penalty="l2", solver="liblinear", cv=2, n_jobs=-1).fit(X_train_list, y_train_list)
                linear_model = sklearn.linear_model.LogisticRegressionCV(scoring='roc_auc').fit(X_train_list, y_train_list)
                tree_model = xgb.XGBClassifier().fit(X_train_list, y_train_list)
            
            y_train_proba = linear_model.predict_proba(X_train_list)[::, 1]
            y_val_proba = linear_model.predict_proba(X_val_list)[::, 1]
            y_train_proba_tree = tree_model.predict_proba(X_train_list)[::, 1]
            y_val_proba_tree = tree_model.predict_proba(X_val_list)[::, 1]
            
            # save y_val_proba_tree
            idx_censored = np.where((y_val[:, idx] == 'Censored'))[0]
            tree_model_proba_save_path = os.path.join(model_save_path, f"tree_model_proba_{epoch}epoch_{model_choice}_{label_col}_{month_date}.pkl")
            y_val_proba_tree = np.array(insert_zeros(y_val_proba_tree, idx_censored))
            with open(tree_model_proba_save_path, 'wb') as f:
                pickle.dump(y_val_proba_tree, f)

            if confidence_interval:
                auroc_train, auroc_val, auroc_train_ci, auroc_val_ci  = run_analysis("Logistic Regression", y_train_list, y_train_proba, y_val_task, y_val_proba, label_col, confidence_interval=True)
                auroc_train_tree, auroc_val_tree, auroc_train_tree_ci, auroc_val_tree_ci = run_analysis("XGBoost", y_train_list, y_train_proba_tree, y_val_task, y_val_proba_tree, label_col, confidence_interval=False)

                auroc_train_dict[f'auroc_train_{label_col}'] = auroc_train
                auroc_val_dict[f'auroc_val_{label_col}'] = auroc_val
                auroc_train_dict[f'auroc_train_tree_{label_col}'] = auroc_train_tree
                auroc_val_dict[f'auroc_val_tree_{label_col}'] = auroc_val_tree
                
                auroc_train_dict[f'auroc_train_ci_{label_col}'] = auroc_train_ci
                auroc_val_dict[f'auroc_val_ci_{label_col}'] = auroc_val_ci
                auroc_train_dict[f'auroc_train_tree_ci_{label_col}'] = auroc_train_tree_ci
                auroc_val_dict[f'auroc_val_tree_ci_{label_col}'] = auroc_val_tree_ci

            else:
                auroc_train, auroc_val = run_analysis("Logistic Regression", y_train_list, y_train_proba, y_val_task, y_val_proba, label_col, confidence_interval=False)
                auroc_train_tree, auroc_val_tree = run_analysis("XGBoost", y_train_list, y_train_proba_tree, y_val_task, y_val_proba_tree, label_col, confidence_interval=False)
                
                auroc_train_dict[f'auroc_train_{label_col}'] = auroc_train
                auroc_val_dict[f'auroc_val_{label_col}'] = auroc_val
                auroc_train_dict[f'auroc_train_tree_{label_col}'] = auroc_train_tree
                auroc_val_dict[f'auroc_val_tree_{label_col}'] = auroc_val_tree
            if model_save_path:
                linear_model_save_path = os.path.join(model_save_path, f"linear_model_{epoch}epoch_{model_choice}_{label_col}_{month_date}.pkl")
                with open(linear_model_save_path,'wb') as f:
                    pickle.dump(linear_model, f)
                tree_model_save_path = os.path.join(model_save_path, f"tree_model_{epoch}epoch_{model_choice}_{label_col}_{month_date}.pkl")
                with open(tree_model_save_path,'wb') as f:
                    pickle.dump(tree_model, f)
            try:    
                auroc_val_list.append(auroc_val_tree)
            except:
                auroc_val_list.append(0)
                auroc_val = 0
        # metric_values.append(np.mean(auroc_val_list))
        
            
    return metric_values, run, writer, auroc_val_tree, auroc_train_dict, auroc_val_dict

# given the feature X in a split, return label df
def get_split_df(image_paths, label_df, X, nii_folder, task_set):
    image_paths = np.array(image_paths)
    patient_datetime_ls = []
    for image_path in image_paths.squeeze():
        if '/local-scratch-nvme/nigam/PE/anon_nii_gz/anon_nii_gz' in image_path:
            image_path = image_path.replace('/local-scratch-nvme/nigam/PE/anon_nii_gz/anon_nii_gz', '/local-scratch/nigam/datasets/PE/inspect/anon_nii_gz/anon_nii_gz')
        
        # patient_datetime = image_path.replace(nii_folder, '').replace('.nii.gz', '').replace('/', '')
        patient_datetime = image_path.replace(nii_folder, '').replace('/local-scratch/nigam/datasets/PE/inspect/anon_nii_gz/anon_nii_gz', '').replace('.nii.gz', '').replace('/', '')
        patient_id, date, time = patient_datetime.split('_', 2)
        patient_datetime = patient_id+'_'+date+'T'+time.replace('_', ':')
        patient_datetime_ls.append(patient_datetime)

    row_ls = []
    for i, patient_datetime in enumerate(patient_datetime_ls):
        try:
            labels_columns = []
            for task in task_set:
                labels_columns.extend([f'tte_{task}', f'is_censored_{task}'])
            row = label_df.loc[label_df['patient_datetime'] == patient_datetime][labels_columns].iloc[0, :]
            row_ls.append(row)
        except:
            print(f'patient_datetime {patient_datetime} not found in label_df, probs negative tte readmission, PH')
            X = np.delete(X, i, axis=0)
    
    return pd.concat(row_ls, axis=1).T, X

def survival_probe(task_set, X_train, image_paths_train,  X_val, image_paths_val, X_test, image_paths_test, label_df, nii_folder, confidence_interval=True):
    label_df_train, X_train = get_split_df(image_paths_train, label_df, X_train, nii_folder, task_set)
    label_df_test, X_test = get_split_df(image_paths_test, label_df, X_test, nii_folder, task_set)
    label_df_val, X_val = get_split_df(image_paths_val, label_df, X_val, nii_folder, task_set)
    
    tdcs_train_dict, tdcs_test_dict = {} , {}
    ibs_train_dict, ibs_test_dict = {} , {}
    
    for task in task_set:
        y_train = label_df_train[f'tte_{task}'].astype(np.int32).to_numpy(), label_df_train[f'is_censored_{task}'].astype(bool).to_numpy()
        y_test = label_df_test[f'tte_{task}'].astype(np.int32).to_numpy(), label_df_test[f'is_censored_{task}'].astype(bool).to_numpy()
        y_val = label_df_val[f'tte_{task}'].astype(np.int32).to_numpy(), label_df_val[f'is_censored_{task}'].astype(bool).to_numpy()

        in_features = X_train.shape[1]
        num_nodes = [32, 32]
        out_features = 1
        dropout = 0.1
        output_bias = False
        batch_size = 128
        lr = 1e-3
        epochs = 3
        val = X_val, y_val
        batch_norm = True
        
        (batch_size, num_nodes, dropout, lr) = (128, [256, 256], 0.1, 0.0001)
        print(f'batch_size= {batch_size}, num_nodes= {num_nodes}, dropout= {dropout}, lr= {lr}')
        net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                                    dropout, output_bias=output_bias)
        model = CoxPH(net, tt.optim.Adam)

        model.optimizer.set_lr(lr)

        callbacks = [tt.callbacks.EarlyStopping()]
        verbose = False

        log = model.fit(X_train[:-1], (y_train[0][:-1], y_train[1][:-1]), batch_size, epochs, callbacks, verbose, val_data=val, val_batch_size=batch_size)

        hazards= model.predict(X_train)
        tdcs = test_c_statistic(y_train[0], ~y_train[1], hazards)
        tdcs_train_dict[f'tdcs_train_{task}'] = tdcs
        
        _ = model.compute_baseline_hazards()
        surv = model.predict_surv_df(X_train)
        ev = EvalSurv(surv, y_train[0], y_train[1], censor_surv='km')
        time_grid = np.linspace(y_train[0].min(), y_train[0].max(), 100)
        ibs_train_dict[f'ibs_train_{task}'] = ev.integrated_brier_score(time_grid)
        
        hazards= model.predict(X_test)
        tdcs = test_c_statistic(y_test[0], ~y_test[1], hazards)
        tdcs_test_dict[f'tdcs_test_{task}'] = tdcs
        print('test tdcs', tdcs_test_dict[f'tdcs_test_{task}'])
        
        surv = model.predict_surv_df(X_test)
        ev = EvalSurv(surv, y_test[0], y_test[1], censor_surv='km')
        time_grid = np.linspace(y_test[0].min(), y_test[0].max(), 100)
        ibs_test_dict[f'ibs_test_{task}'] = ev.integrated_brier_score(time_grid)
        print('test ibs', ibs_test_dict[f'ibs_test_{task}'])
        
        if confidence_interval:
            tdcs_test_dict[f'tdcs_test_ci_{task}'] = bootstrap_tdcs(y_test[0], ~y_test[1], hazards)
            print('test tdcs ci', tdcs_test_dict[f'tdcs_test_ci_{task}'])
            
            ibs_test_dict[f'ibs_test_ci_{task}'] = bootstrap_ibs(model, X_test, y_test)
            print('test ibs ci', ibs_test_dict[f'ibs_test_ci_{task}'])
        
    return tdcs_train_dict, tdcs_test_dict, ibs_train_dict, ibs_test_dict


def tune_hyperparameter_tree(tree_model, X_val, y_val):
    param_grid = {
                'max_depth': [3, 6], # , 10
                'learning_rate': [0.01, 0.1, 0.2],
                'n_estimators': [100, 500], # , 1000
                'colsample_bytree': [0.3, 0.7]
            }
    grid_search = GridSearchCV(estimator=tree_model, param_grid=param_grid, scoring='roc_auc', n_jobs=-1, cv=3, verbose=1)

    grid_search.fit(X_val, y_val)
    best_model = grid_search.best_estimator_
    return best_model


def tune_hyperparameter_LR(linear_model, X_val, y_val):
    param_grid = {
    'penalty': ['l2', 'elasticnet'], # 'l1' , 'none'
    #'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    }

    # Setup GridSearchCV
    grid_search = GridSearchCV(estimator=linear_model, param_grid=param_grid, scoring='roc_auc', n_jobs=-1, cv=5, verbose=1)

    # Fit the model
    grid_search.fit(X_val, y_val)
    
    best_model = grid_search.best_estimator_  # or random_search.best_estimator_

    return best_model

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


def insert_zeros(numbers, indices):
    new_list = []
    i = 0
    total_length = len(numbers) + len(indices)
    indices_set = set(indices)  # Convert to set for faster lookup

    for j in range(total_length):
        if j in indices_set:
            new_list.append(0)
        else:
            new_list.append(numbers[i])
            i += 1
    return new_list


import warnings
import numpy as np
import pandas as pd
from pycox.evaluation.concordance import concordance_td
from pycox.evaluation import ipcw, admin
from pycox import utils


class EvalSurv:
    """Class for evaluating predictions.
    
    Arguments:
        surv {pd.DataFrame} -- Survival predictions.
        durations {np.array} -- Durations of test set.
        events {np.array} -- Events of test set.

    Keyword Arguments:
        censor_surv {str, pd.DataFrame, EvalSurv} -- Censoring distribution.
            If provided data frame (survival function for censoring) or EvalSurv object,
            this will be used. 
            If 'km', we will fit a Kaplan-Meier to the dataset.
            (default: {None})
        censor_durations {np.array}: -- Administrative censoring times. (default: {None})
        steps {str} -- For durations between values of `surv.index` choose the higher index 'pre'
            or lower index 'post'. For a visualization see `help(EvalSurv.steps)`. (default: {'post'})
    """
    def __init__(self, surv, durations, events, censor_surv=None, censor_durations=None, steps='post'):
        assert (type(durations) == type(events) == np.ndarray), 'Need `durations` and `events` to be arrays'
        self.surv = surv
        self.durations = durations
        self.events = events
        self.censor_surv = censor_surv
        self.censor_durations = censor_durations
        self.steps = steps
        assert pd.Series(self.index_surv).is_monotonic_increasing

    @property
    def censor_surv(self):
        """Estimated survival for censorings. 
        Also an EvalSurv object.
        """
        return self._censor_surv

    @censor_surv.setter
    def censor_surv(self, censor_surv):
        if isinstance(censor_surv, EvalSurv):
            self._censor_surv = censor_surv
        elif type(censor_surv) is str:
            if censor_surv == 'km':
                self.add_km_censor()
            else:
                raise ValueError(f"censor_surv cannot be {censor_surv}. Use e.g. 'km'")
        elif censor_surv is not None:
            self.add_censor_est(censor_surv)
        else:
            self._censor_surv = None

    @property
    def index_surv(self):
        return self.surv.index.values

    @property
    def steps(self):
        """How to handle predictions that are between two indexes in `index_surv`.

        For a visualization, run the following:
            ev = EvalSurv(pd.DataFrame(np.linspace(1, 0, 7)), np.empty(7), np.ones(7), steps='pre')
            ax = ev[0].plot_surv()
            ev.steps = 'post'
            ev[0].plot_surv(ax=ax, style='--')
            ax.legend(['pre', 'post'])
        """
        return self._steps

    @steps.setter
    def steps(self, steps):
        vals = ['post', 'pre']
        if steps not in vals:
            raise ValueError(f"`steps` needs to be {vals}, got {steps}")
        self._steps = steps

    def add_censor_est(self, censor_surv, steps='post'):
        """Add censoring estimates so one can use inverse censoring weighting.
        `censor_surv` are the survival estimates trained on (durations, 1-events),
        
        Arguments:
            censor_surv {pd.DataFrame} -- Censor survival curves.

    Keyword Arguments:
        round {str} -- For durations between values of `surv.index` choose the higher index 'pre'
            or lower index 'post'. If `None` use `self.steps` (default: {None})
        """
        if not isinstance(censor_surv, EvalSurv):
            censor_surv = self._constructor(censor_surv, self.durations, 1-self.events, None,
                                            steps=steps)
        self.censor_surv = censor_surv
        return self

    def add_km_censor(self, steps='post'):
        """Add censoring estimates obtained by Kaplan-Meier on the test set
        (durations, 1-events).
        """
        km = utils.kaplan_meier(self.durations, 1-self.events)
        surv = pd.DataFrame(np.repeat(km.values.reshape(-1, 1), len(self.durations), axis=1),
                            index=km.index)
        return self.add_censor_est(surv, steps)

    @property
    def censor_durations(self):
        """Administrative censoring times."""
        return self._censor_durations
    
    @censor_durations.setter
    def censor_durations(self, val):
        if val is not None:
            assert (self.durations[self.events == 0] == val[self.events == 0]).all(),\
                'Censored observations need same `durations` and `censor_durations`'
            assert (self.durations[self.events == 1] <= val[self.events == 1]).all(),\
                '`durations` cannot be larger than `censor_durations`'
            if (self.durations == val).all():
                warnings.warn("`censor_durations` are equal to `durations`." +
                              " `censor_durations` are likely wrong!")
            self._censor_durations = val
        else:
            self._censor_durations = val

    @property
    def _constructor(self):
        return EvalSurv

    def __getitem__(self, index):
        if not (hasattr(index, '__iter__') or type(index) is slice) :
            index = [index]
        surv = self.surv.iloc[:, index]
        durations = self.durations[index]
        events = self.events[index]
        new = self._constructor(surv, durations, events, None, steps=self.steps)
        if self.censor_surv is not None:
            new.censor_surv = self.censor_surv[index]
        return new

    def plot_surv(self, **kwargs):
        """Plot survival estimates. 
        kwargs are passed to `self.surv.plot`.
        """
        if len(self.durations) > 50:
            raise RuntimeError("We don't allow to plot more than 50 lines. Use e.g. `ev[1:5].plot()`")
        if 'drawstyle' in kwargs:
            raise RuntimeError(f"`drawstyle` is set by `self.steps`. Remove from **kwargs")
        return self.surv.plot(drawstyle=f"steps-{self.steps}", **kwargs)

    def idx_at_times(self, times):
        """Get the index (iloc) of the `surv.index` closest to `times`.
        I.e. surv.loc[tims] (almost)= surv.iloc[idx_at_times(times)].

        Useful for finding predictions at given durations.
        """
        return utils.idx_at_times(self.index_surv, times, self.steps)

    def _duration_idx(self):
        return self.idx_at_times(self.durations)

    def surv_at_times(self, times):
        idx = self.idx_at_times(times)
        return self.surv.iloc[idx]

    # def prob_alive(self, time_grid):
    #     return self.surv_at_times(time_grid).values

    def concordance_td(self, method='adj_antolini'):
        """Time dependent concorance index from
        Antolini, L.; Boracchi, P.; and Biganzoli, E. 2005. A time-dependent discrimination
        index for survival data. Statistics in Medicine 24:3927–3944.

        If 'method' is 'antolini', the concordance from Antolini et al. is computed.
    
        If 'method' is 'adj_antolini' (default) we have made a small modifications
        for ties in predictions and event times.
        We have followed step 3. in Sec 5.1. in Random Survival Forests paper, except for the last
        point with "T_i = T_j, but not both are deaths", as that doesn't make much sense.
        See 'metrics._is_concordant'.

        Keyword Arguments:
            method {str} -- Type of c-index 'antolini' or 'adj_antolini' (default {'adj_antolini'}).

        Returns:
            float -- Time dependent concordance index.
        """
        return concordance_td(self.durations, self.events, self.surv.values,
                              self._duration_idx(), method)

    def brier_score(self, time_grid, max_weight=np.inf):
        """Brier score weighted by the inverse censoring distribution.
        See Section 3.1.2 or [1] for details of the wighting scheme.
        
        Arguments:
            time_grid {np.array} -- Durations where the brier score should be calculated.

        Keyword Arguments:
            max_weight {float} -- Max weight value (max number of individuals an individual
                can represent (default {np.inf}).

        References:
            [1] Håvard Kvamme and Ørnulf Borgan. The Brier Score under Administrative Censoring: Problems
                and Solutions. arXiv preprint arXiv:1912.08581, 2019.
                https://arxiv.org/pdf/1912.08581.pdf
        """
        if self.censor_surv is None:
            raise ValueError("""Need to add censor_surv to compute Brier score. Use 'add_censor_est'
            or 'add_km_censor' for Kaplan-Meier""")
        bs = ipcw.brier_score(time_grid, self.durations, self.events, self.surv.values,
                              self.censor_surv.surv.values, self.index_surv,
                              self.censor_surv.index_surv, max_weight, True, self.steps,
                              self.censor_surv.steps)
        return pd.Series(bs, index=time_grid).rename('brier_score')

    def nbll(self, time_grid, max_weight=np.inf):
        """Negative binomial log-likelihood weighted by the inverse censoring distribution.
        See Section 3.1.2 or [1] for details of the wighting scheme.
        
        Arguments:
            time_grid {np.array} -- Durations where the brier score should be calculated.

        Keyword Arguments:
            max_weight {float} -- Max weight value (max number of individuals an individual
                can represent (default {np.inf}).

        References:
            [1] Håvard Kvamme and Ørnulf Borgan. The Brier Score under Administrative Censoring: Problems
                and Solutions. arXiv preprint arXiv:1912.08581, 2019.
                https://arxiv.org/pdf/1912.08581.pdf
        """
        if self.censor_surv is None:
            raise ValueError("""Need to add censor_surv to compute the score. Use 'add_censor_est'
            or 'add_km_censor' for Kaplan-Meier""")
        bll = ipcw.binomial_log_likelihood(time_grid, self.durations, self.events, self.surv.values,
                                           self.censor_surv.surv.values, self.index_surv,
                                           self.censor_surv.index_surv, max_weight, True, self.steps,
                                           self.censor_surv.steps)
        return pd.Series(-bll, index=time_grid).rename('nbll')

    def integrated_brier_score(self, time_grid, max_weight=np.inf):
        """Integrated Brier score weighted by the inverse censoring distribution.
        Essentially an integral over values obtained from `brier_score(time_grid, max_weight)`.
        
        Arguments:
            time_grid {np.array} -- Durations where the brier score should be calculated.

        Keyword Arguments:
            max_weight {float} -- Max weight value (max number of individuals an individual
                can represent (default {np.inf}).
        """
        if self.censor_surv is None:
            raise ValueError("Need to add censor_surv to compute briser score. Use 'add_censor_est'")
        return ipcw.integrated_brier_score(time_grid, self.durations, self.events, self.surv.values,
                                           self.censor_surv.surv.values, self.index_surv,
                                           self.censor_surv.index_surv, max_weight, self.steps,
                                           self.censor_surv.steps)

    def integrated_nbll(self, time_grid, max_weight=np.inf):
        """Integrated negative binomial log-likelihood weighted by the inverse censoring distribution.
        Essentially an integral over values obtained from `nbll(time_grid, max_weight)`.
        
        Arguments:
            time_grid {np.array} -- Durations where the brier score should be calculated.

        Keyword Arguments:
            max_weight {float} -- Max weight value (max number of individuals an individual
                can represent (default {np.inf}).
        """
        if self.censor_surv is None:
            raise ValueError("Need to add censor_surv to compute the score. Use 'add_censor_est'")
        ibll = ipcw.integrated_binomial_log_likelihood(time_grid, self.durations, self.events, self.surv.values,
                                                       self.censor_surv.surv.values, self.index_surv,
                                                       self.censor_surv.index_surv, max_weight, self.steps,
                                                       self.censor_surv.steps)
        return -ibll

    def brier_score_admin(self, time_grid):
        """The Administrative Brier score proposed by [1].
        Removes individuals as they are administratively censored, event if they have experienced an
        event. 
        
        Arguments:
            time_grid {np.array} -- Durations where the brier score should be calculated.

        References:
            [1] Håvard Kvamme and Ørnulf Borgan. The Brier Score under Administrative Censoring: Problems
                and Solutions. arXiv preprint arXiv:1912.08581, 2019.
                https://arxiv.org/pdf/1912.08581.pdf
        """
        if self.censor_durations is None:
            raise ValueError("Need to provide `censor_durations` (censoring durations) to use this method")
        bs = admin.brier_score(time_grid, self.durations, self.censor_durations, self.events,
                               self.surv.values, self.index_surv, True, self.steps)
        return pd.Series(bs, index=time_grid).rename('brier_score')

    def integrated_brier_score_admin(self, time_grid):
        """The Integrated administrative Brier score proposed by [1].
        Removes individuals as they are administratively censored, event if they have experienced an
        event. 
        
        Arguments:
            time_grid {np.array} -- Durations where the brier score should be calculated.

        References:
            [1] Håvard Kvamme and Ørnulf Borgan. The Brier Score under Administrative Censoring: Problems
                and Solutions. arXiv preprint arXiv:1912.08581, 2019.
                https://arxiv.org/pdf/1912.08581.pdf
        """
        if self.censor_durations is None:
            raise ValueError("Need to provide `censor_durations` (censoring durations) to use this method")
        ibs = admin.integrated_brier_score(time_grid, self.durations, self.censor_durations, self.events,
                                           self.surv.values, self.index_surv, self.steps)
        return ibs

    def nbll_admin(self, time_grid):
        """The negative administrative binomial log-likelihood proposed by [1].
        Removes individuals as they are administratively censored, event if they have experienced an
        event. 
        
        Arguments:
            time_grid {np.array} -- Durations where the brier score should be calculated.

        References:
            [1] Håvard Kvamme and Ørnulf Borgan. The Brier Score under Administrative Censoring: Problems
                and Solutions. arXiv preprint arXiv:1912.08581, 2019.
                https://arxiv.org/pdf/1912.08581.pdf
        """
        if self.censor_durations is None:
            raise ValueError("Need to provide `censor_durations` (censoring durations) to use this method")
        bll = admin.binomial_log_likelihood(time_grid, self.durations, self.censor_durations, self.events,
                                           self.surv.values, self.index_surv, True, self.steps)
        return pd.Series(-bll, index=time_grid).rename('nbll')

    def integrated_nbll_admin(self, time_grid):
        """The Integrated negative administrative binomial log-likelihood score proposed by [1].
        Removes individuals as they are administratively censored, event if they have experienced an
        event. 
        
        Arguments:
            time_grid {np.array} -- Durations where the brier score should be calculated.

        References:
            [1] Håvard Kvamme and Ørnulf Borgan. The Brier Score under Administrative Censoring: Problems
                and Solutions. arXiv preprint arXiv:1912.08581, 2019.
                https://arxiv.org/pdf/1912.08581.pdf
        """
        if self.censor_durations is None:
            raise ValueError("Need to provide `censor_durations` (censoring durations) to use this method")
        ibll = admin.integrated_binomial_log_likelihood(time_grid, self.durations, self.censor_durations,
                                                        self.events, self.surv.values, self.index_surv,
                                                        self.steps)
        return -ibll
    

    
    
def test_c_statistic(times, is_censor, hazards):
    # N = 200
    # B = 8

    # END = 40

    # np.random.seed(12313)

    # time_bins = np.linspace(0, END * 0.8, num=B).astype(np.float64)
    time_bins = [0, np.inf]

    # times = np.random.randint(0, END, size=(N,)).astype(np.float64)
    # is_censor = np.random.binomial(1, 0.7, size=(N,)).astype(bool)
    # hazards = np.random.normal(size=(N, B)).astype(np.float64)

    # for i in range(N):
    #     if times[i] > 10 and not is_censor[i]:
    #         hazards[i, :] += 1
    #     if times[i] > 50 and not is_censor[i]:
    #         hazards[i, :] += 3

    total_auroc = 0

    surv = 1

    total_weight = 0

    for time in sorted(list(set(times))):
        mask = (((times > time) & (is_censor != 0)) | ((times >= time) & (is_censor == 0))).astype(bool)

        for i, val in enumerate(time_bins):
            if time < val:
                i -= 1
                break

        risks = hazards[:, i]

        correct = (times == time) & (is_censor == 0)

        f = correct.sum() / mask.sum()
        factor = surv * f
        surv *= 1 - f

        weight = factor * surv

        if correct.sum() == mask.sum() or correct.sum() == 0:
            # assert weight == 0
            pass
        else:
            current_risks = risks[mask]
            current_correct = correct[mask]

            auroc = sklearn.metrics.roc_auc_score(current_correct, current_risks)

            total_auroc += auroc * weight
            total_weight += weight

    expected = total_auroc / total_weight
    
    return expected

    # actual = femr.metrics.compute_c_statistic(times, is_censor, time_bins, hazards)[0]

    # assert actual == expected
    
    
# Function to check for np.nan
def is_nan(x):
    try:
        return np.isnan(x)
    except TypeError:
        return False
    
import torch.distributed.rpc as rpc
# Initialize the RPC framework
def setup_rpc(rank, world_size):
    backend = "gloo"  # or "tensorpipe", depending on what you're using
    print(f"Rank: {rank}, World Size: {world_size}, Backend: {backend}")
    
    try:
        rpc.init_rpc(
            name=f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                num_worker_threads=8  # or whatever options you are using
            )
        )
    except Exception as e:
        print(f"Failed to initialize RPC. Error: {e}")
        raise
    
    
def cleanup_rpc():
    rpc.shutdown()


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Args:
            patience (int): How many epochs to wait after the last improvement.
            min_delta (float): The minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0