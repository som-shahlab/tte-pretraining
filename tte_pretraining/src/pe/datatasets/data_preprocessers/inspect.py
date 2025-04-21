import os
import h5py
from tqdm import tqdm
from typing      import Optional
from totalsegmentator.dicom_io import dcm_to_nifti
import dicom2nifti
import nibabel as nib
import fnmatch
import pdb
import concurrent.futures

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import List,Dict
import time
import threading
from concurrent.futures import ProcessPoolExecutor


from pe.datatasets.data_preprocessers.data_preprocessing import  DataPreprocessing
from pe.utils.statistics import get_statistics
from pe.utils.crop_utils import get_bbox_from_mask,crop_to_bbox


from dotenv import load_dotenv
import csv
load_dotenv()

import sys

sys.path.append("/share/pi/nigam/alejandro/Repositories/TotalSegmentatorV2/totalsegmentator")

from python_api import totalsegmentator
#from totalsegmentator.python_api import totalsegmentator

import faulthandler
faulthandler.enable()




def directory_not_exists(directory_path):
    """
    Check if a directory does not exist.

    Args:
    directory_path (str): The path to the directory to check.

    Returns:
    bool: True if the directory does not exist, False otherwise.
    """
    return not os.path.exists(directory_path)

def process_file_parallel_(file) -> str:

        if file.endswith('.json'):
            return ("json",file)

        elif file.endswith('.nii.gz'):
        
            return ("nifti",file)

        else:
            print(f'File {file} ignored')
        


class InspectPreprocessing(DataPreprocessing):
    """
    A class for preprocessing RSNA dataset for medical image analysis.

    Parameters:
    - dataset_path (str): The path to the dataset.
    """

    def __init__(self,dataset_path:str,split:str=None,verbose:bool=True):
        """
        Initialize RSNADataPreprocessing with the path to the dataset.

        Parameters:
        - dataset_path (str): The path to the dataset.
        """
        self.verbose      = verbose
        self.dataset_path = dataset_path
        self.json_files   = []
        self.nifti_files  = []
        self.split        = split
        self.debug        = True


        if split: 
            self.directory_path:str = os.path.join(self.dataset_path,split)

        else:
            self.directory_path:str = self.dataset_path

        self.directory_contents     =   os.listdir(self.directory_path)
        self.len_directory_contents  = len(self.directory_contents )

        if verbose:
            print(f"Preprocessing: {self.len_directory_contents} files ")



    def process_file_(self, file) -> None:
        if file.endswith('.json'):
            self.json_files.append(file)

        elif file.endswith('.nii.gz'):
        
            self.nifti_files.append(file)

        else:
            print(f'File {file} ignored')
        



    def process_files_parallel(self, max_workers:int=10,start:int=0,end:int=-1):

        if end > self.len_directory_contents:
            end = -1

        subset_files = self.directory_contents[start:end]
        subset_size  = len(self.directory_contents)
        start_time   = time.time()

        ### Separeate Json and nifti files ####
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for type_,file in executor.map(process_file_parallel_, subset_files):
                if type_ == "json":
                    self.json_files.append(file)

                if type_ == "nifti":
                    self.nifti_files.append(file)

        end_time = time.time()

        # Extrapolate the total time based on the subset
        total_files    = len(self.directory_contents)
        estimated_time = (end_time - start_time)

        if self.verbose:
            print(f"Orignal number of files: {len(subset_files) }")
            print(f"Processed {len(self.json_files)} JSON files")
            print(f"Processed {len(self.nifti_files)} NII files")
            print(f"Total time: {estimated_time:.2f} seconds")

      
        if self.debug:
            pdb.set_trace()

        return  {"json": self.json_files, "nifti":self.nifti_files}

    
    
    def process_files(self,start:int=0,end:int=-1) -> Dict[str,List[str]]:
        """
        Read a split of the RSNA dataset from a CSV file.

        Parameters:
        - debug: bool 

        Returns:
        - data_dict: return a ditctionary contiang "json" (labels) and images (nifit)
        """

        if end > self.len_directory_contents:
            end = -1

        subset_files = self.directory_contents[start:end]
        subset_size  = len(self.directory_contents)

        start_time   = time.time()

        for i,file in enumerate(subset_files):
            self.process_file_(file)

        end_time       = time.time()
        estimated_time = (end_time - start_time)

        if self.verbose:
            print(f"Orignal number of files: {len(subset_files) }")
            print(f"Processed {len(self.json_files)} JSON files")
            print(f"Processed {len(self.nifti_files)} NII files")
            print(f"Total time: {estimated_time:.2f} seconds")

      
        #if self.debug:
        #    pdb.set_trace()
       
        return  {"json": self.json_files, "nifti": self.nifti_files}
    
     
    

    def segment_study_(self,input_nifti_path,outtput_nifti_path,device="gpu"):
        input_path:   os.PathLike   = os.path.join(self.directory_path,input_nifti_path)
        mask_path:    os.PathLike   = os.path.join(self.directory_path,"masks",input_nifti_path)
        output_path:  os.PathLike   = os.path.join(outtput_nifti_path,input_nifti_path.strip('.nii.gz'))

        #pdb.set_trace()
        if directory_not_exists(output_path):
            try:
                #totalsegmentator(input=input_path,output= output_path , preview=True, device="gpu", task = "lung_vessels")
                totalsegmentator(input     = input_path,
                                 output    = output_path , 
                                 crop_path = mask_path,
                                 device    = "gpu",
                                 body_seg  = True,
                                 fastest   = True,
                                 preview   = False, 
                                 quiet     = True)
                                 #
                #totalsegmentator(input=input_path,output= output_path , preview=True, device="gpu",fastest=True)
            except:
                return input_path
        else:
            print(f"Mask for {output_path} has alredy been created")

        return True

    
    def segment_studies(self,save_to:str,start:int=0,end:int=-1):
        """
        Load a study from the RSNA dataset

        Parameters:
        - split (str): The split of the dataset.
        - df (pd.DataFrame): The DataFrame containing the data.
        - study_uid (str): The StudyInstanceUID of the study.
        - extension (str, optional): The file extension of DICOM files (default: '.dcm').

        Returns:
        - dict: A dictionary containing 'volume' (3D NumPy array) and 'labels' (study data as NumPy array).
        """


        print(f"Loading Model From: {os.getenv('TOTALSEG_WEIGHTS_PATH')}" )
    
        
        if self.verbose:
            print("*"*10)
            print("Loading Directoiress..")
            print("*"*10)

        data_dir        = self.process_files(start=int(start),end=int(end))
        missing_nifitis = []


        if self.verbose:
            print("*"*10)
            print("Calculating masks..")
            print("*"*10)
        
        for nifti_path in tqdm(data_dir["nifti"] , total=self.len_directory_contents//2):
            sucsses = self.segment_study_(input_nifti_path=nifti_path,outtput_nifti_path=save_to)


            if sucsses != True:
                missing_nifitis.append(sucsses)

        

        with open('missing_nifti.csv', 'w', newline='') as csv_file:
            csv_out = csv.writer(csv_file)
            csv_out.writerows([missing_nifitis[index]] for index in range(0, len(missing_nifitis)))
               

            #pdb.set_trace()



#if __name__ == '__main__':

    
        
    

