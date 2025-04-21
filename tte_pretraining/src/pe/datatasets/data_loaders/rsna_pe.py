import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional

class RSNAPE(Dataset):
    def __init__(self, file_path:str,data_str:Optional[str]="volume",label_str:Optional[str]="labels",mask_str:Optional[str]="masks"):
        self.file_path = file_path
        self.data      = None
        self.labels    = None
        self.data_str  = data_str
        self.label_str = label_str
        self.mask_str  = mask_str
        self.load_data()

    def load_data(self):
        with h5py.File(self.file_path, 'r') as h5file:
            self.data   = h5file[self.data_str][:]
            self.labels = h5file[self.label_str][:]

            if self.mask_str  != None:
                self.masks = h5file[self.mask_str][:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'data':  self.data[idx],
            'label': self.labels[idx]
        }

        if self.mask_str != None:
            sample["mask"] =  self.masks[idx]

        return sample

