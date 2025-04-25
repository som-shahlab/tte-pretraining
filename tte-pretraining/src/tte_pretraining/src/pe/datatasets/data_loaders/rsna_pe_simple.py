import os
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader

class NiftiDataset(Dataset):
    def __init__(self, csv_file, root_dir,split:str="train", transform=None):
        self.metadata_file  = pd.read_csv(csv_file)
        self.root_dir       = root_dir
        self.transform      = transform
        df                  = pd.read_csv(csv_file)

        # Filter out studies not present in root_dir
        available_studies = set(os.listdir(os.path.join(root_dir,split)))
        self.metadata        = []
        for index, row in df.iterrows():
            if row["study_uid"] in available_studies:
                self.metadata.append([os.path.join(root_dir, row["split"],row["study_uid"] ,row["series_uid"] , 'volume.nii.gz'),row["label"]])
               

        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        nifti_data = nib.load(self.metadata[idx][0]).get_fdata()
        label      = self.metadata[idx][1]

        #sample = {'nifti_data': nifti_data, 'label': label}
        sample = [nifti_data,label]

        if self.transform:
            #sample['nifti_data'] = self.transform(sample['nifti_data'])
            sample[0] = self.transform(sample[0])

        return sample
