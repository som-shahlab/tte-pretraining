import os
import h5py
import numpy  as np
import pandas as pd
import pydicom
from tqdm import tqdm
from typing      import Optional
from totalsegmentator.dicom_io import dcm_to_nifti
import dicom2nifti
import nibabel as nib
import fnmatch

import gzip
from dataclasses import dataclass
from pathlib import Path


from pe.datatasets.data_preprocessers.data_preprocessing import  DataPreprocessing
from pe.utils.statistics import get_statistics
from pe.utils.crop_utils import get_bbox_from_mask,crop_to_bbox

from totalsegmentator.python_api import totalsegmentator
from dotenv import load_dotenv
load_dotenv()



def directory_not_exists(directory_path):
    """
    Check if a directory does not exist.

    Args:
    directory_path (str): The path to the directory to check.

    Returns:
    bool: True if the directory does not exist, False otherwise.
    """
    return not os.path.exists(directory_path)


@dataclass
class DicomItem:
    """
    Class for representing a DICOM item in an inventory.

    This class is used to store information about a DICOM item, including its pixel array, index, exposure, and thickness.

    Parameters:
    - array (float): The pixel array of the DICOM item.
    - index (int): The index of the DICOM item.
    - exposure (float): The exposure value of the DICOM item.
    - thickness (int): The thickness of the DICOM item in millimeters.
    """

    array: float
    index: int
    exposure: float
    thickness: int
 



class RSNAPePreprocessing(DataPreprocessing):
    """
    A class for preprocessing RSNA dataset for medical image analysis.

    Parameters:
    - dataset_path (str): The path to the dataset.
    """
    def __init__(self,dataset_path:str):
        """
        Initialize RSNADataPreprocessing with the path to the dataset.

        Parameters:
        - dataset_path (str): The path to the dataset.
        """
        self.dataset_path = dataset_path
    
    def read_splits_(self,split:str='train',engine:Optional[str]="pyarrow") -> pd.DataFrame:
        """
        Read a split of the RSNA dataset from a CSV file.

        Parameters:
        - split (str): The split to read (e.g., 'train').
        - engine (str, optional): The CSV engine to use for reading (default: 'pyarrow').

        Returns:
        - pd.DataFrame: A DataFrame containing the split data.
        """
        df      = pd.read_csv(os.path.join(self.dataset_path,split+".csv"),engine="pyarrow")
        return df
    
    @staticmethod
    def get_unique_studies_(df:pd.DataFrame,studies_colunm:str="StudyInstanceUID") -> list:
        """
        Get a list of unique study IDs from a DataFrame.

        Parameters:
        - df (pd.DataFrame): The DataFrame containing the data.
        - studies_column (str, optional): The column name containing study IDs (default: 'StudyInstanceUID').

        Returns:
        - list of str: A list of unique study IDs.
        """
        return df[studies_colunm].unique()

    @staticmethod
    def load_dicom_(path:str) -> DicomItem:
        """
        Load a DICOM file and convert it into a DicomItem.

        Parameters:
        - path (str): The path to the DICOM file.

        Returns:
        - DicomItem: A DicomItem object containing DICOM information.
        """
        dicom        = pydicom.dcmread(path)
        dicom_object = DicomItem(array=dicom.pixel_array,index=dicom.InstanceNumber,exposure=dicom.Exposure,thickness=dicom.SliceThickness )
    
     
        return dicom_object

    @staticmethod
    def creat_volume(dicoms):
        """
        Create a 3D volume from a list of DICOM arrays.

        Parameters:
        - dicoms: A list of DICOM arrays.

        Returns:
        - np.ndarray: A 3D NumPy array representing the volume.
        """
        volume = []
        for dicom in dicoms:
            volume.append(dicom.array)

        volume = np.array(volume)

        return  volume
    
    @staticmethod
    def pad_array_along_n_dimension(arr, c, pad_value=0,pad_option=3):
        """
        Pad a 3D NumPy array along its first dimension.

        Parameters:
        - arr (numpy.ndarray): The input array of shape (n, m, z).
        - c (int): The number of elements to pad along the first dimension.
        - pad_value (int or float, optional): The value to use for padding (default: 0).

        Returns:
        - numpy.ndarray: The padded array of shape (n + c, m, z).
        """

        if c <= 0:
            return arr  # No padding needed
        
        # Create a tuple of padding configurations (before, after) for each dimension
        if pad_option == 3:
            padding = ((c, 0), (0, 0), (0, 0))
        elif pad_option == 2:
            padding = ((c, 0), (0, 0))

        
        # Pad the array along the first dimension with the specified value
        padded_array = np.pad(arr, padding, mode='constant', constant_values=pad_value)
        
        return padded_array

    def load_study_(self,split:str,df:pd.DataFrame,study_uid:str,extension:Optional[str]=".dcm",maximum_pad:int=145):
        study_df       = df[ df["StudyInstanceUID"] == study_uid]
        dicoms         = []

        ## Load dicom ###
        for index , study in study_df.iterrows():
            assert study["StudyInstanceUID"]   == study_uid
            path         = os.path.join(self.dataset_path,split,study_uid,study["SeriesInstanceUID"],study["SOPInstanceUID"]+ extension)
            dicom_object = self.load_dicom_(path=path)
            dicoms.append(dicom_object)

        study_df.pop("StudyInstanceUID")
        study_df.pop("SeriesInstanceUID")
        study_df.pop("SOPInstanceUID")

        ## Sort dicom by index ##
        dicoms  = sorted(dicoms, key=lambda x: x.index, reverse=False)
        volume  = self.creat_volume(dicoms)
        labels  = study_df.to_numpy()
        labels  = labels.astype(np.int8)

      

        ### pad ###
        volume_shape = volume.shape
        z            = volume_shape[0]
        p            = maximum_pad - z
        volume       = self.pad_array_along_n_dimension(volume, p, pad_value=0,pad_option=3)
        labels       = self.pad_array_along_n_dimension(labels ,p, pad_value=0,pad_option=2)
        mask         = np.zeros(maximum_pad)
        mask[:z]     = 1

   
        data_point = {"volume":volume ,"labels": labels,"masks":mask,"volume_shape":volume_shape,"thickness": dicom_object.thickness}

        return  data_point
    

       # def load_studies(self,split:str,engine:Optional[str]="pyarrow",studies_colunm:Optional[str]="StudyInstanceUID",extension:Optional[str]=".dcm",save_to:str="post_processed",maximum_pad:int=824):
    #     """
    #     Load a study from the RSNA dataset.

    #     Parameters:
    #     - split (str): The split of the dataset.
    #     - df (pd.DataFrame): The DataFrame containing the data.
    #     - study_uid (str): The StudyInstanceUID of the study.
    #     - extension (str, optional): The file extension of DICOM files (default: '.dcm').

    #     Returns:
    #     - dict: A dictionary containing 'volume' (3D NumPy array) and 'labels' (study data as NumPy array).
    #     """

    #     df                  = self.read_splits_(split=split,engine=engine) 
    #     unique_studies_uid  = self.get_unique_studies_(df=df,studies_colunm=studies_colunm) 
    #     total_tudies        = len(unique_studies_uid)
    #     volumes,volume_shapes,thickness,labels,masks  = [],[],[],[],[]
                  
    #     post_processed_path = os.path.join(self.dataset_path,save_to,split+".h5")

    #     print("Loading Studies")
    #     #i = 0
    #     for study_uid in  tqdm(unique_studies_uid, total=total_tudies):
    #         data_point   = self.load_study_(split=split,df=df,study_uid=study_uid,extension=extension,maximum_pad=maximum_pad)

    #         volumes.append(data_point["volume"])
    #         labels.append(data_point["labels"])
    #         masks.append(data_point["masks"])
    #         volume_shapes.append(data_point["volume_shape"])
    #         thickness.append(data_point["thickness"])
    #         #i += 1
            
    #         #if i  == 2:
    #         #    break
    #     volumes       = np.array(volumes)
    #     labels        = np.array(labels)
    #     masks         = np.array(masks)
    #     volume_shapes = np.array(volume_shapes)
    #     thickness     = np.array(thickness)
            
    #     z,x,y = [],[],[]
    #     for shape in volume_shapes:
    #         z.append(shape[0])
    #         x.append(shape[1])
    #         y.append(shape[2])
    #     z,x,y = np.array(z),np.array(x),np.array(y)

    #     get_statistics(z,os.path.join(self.dataset_path,save_to,f"z_{split}.json"))
    #     get_statistics(x,os.path.join(self.dataset_path,save_to,f"x_{split}.json"))
    #     get_statistics(y,os.path.join(self.dataset_path,save_to,f"y_{split}.json"))
    #     get_statistics(thickness,os.path.join(self.dataset_path,save_to,f"thickness_{split}.json"))



    #     print(f"Saving Studies to {post_processed_path}")
    #     with h5py.File(post_processed_path, 'w') as file:
    #         file.create_dataset('volume',        data=volumes)
    #         file.create_dataset('labels',        data=labels)
    #         file.create_dataset('masks',         data=masks)
    #         file.create_dataset('volume_shape',  data=volume_shapes)
    #         file.create_dataset('thickness',     data= thickness)


    def load_studies(self,split:str,engine:Optional[str]="pyarrow",studies_colunm:Optional[str]="StudyInstanceUID",extension:Optional[str]=".dcm",save_to:str="post_processed",maximum_pad:int=824):
        """
        Load a study from the RSNA dataset.

        Parameters:
        - split (str): The split of the dataset.
        - df (pd.DataFrame): The DataFrame containing the data.
        - study_uid (str): The StudyInstanceUID of the study.
        - extension (str, optional): The file extension of DICOM files (default: '.dcm').

        Returns:
        - dict: A dictionary containing 'volume' (3D NumPy array) and 'labels' (study data as NumPy array).
        """

        df                  = self.read_splits_(split=split,engine=engine) 
        unique_studies_uid  = self.get_unique_studies_(df=df,studies_colunm=studies_colunm) 
        total_tudies        = len(unique_studies_uid)
        post_processed_path = os.path.join(self.dataset_path,save_to,split+".h5")
        collect_list        = ["volume","labels","masks","volume_shape","thickness"]
       
        print("Loading Studies")
                
        for index, study_uid in tqdm(enumerate(unique_studies_uid), total=total_tudies):
            data_point = self.load_study_(split=split, df=df, study_uid=study_uid, extension=extension, maximum_pad=maximum_pad)

            if  index == 0 :
                with h5py.File(post_processed_path, 'w') as file:
                    for collect_item_name in collect_list:
                        collected_item = np.array(data_point[collect_item_name])
                        file.create_dataset(collect_item_name, data=collected_item, compression="gzip", chunks=True)
        

            else:
                with h5py.File(post_processed_path, 'a') as file:
                    for collect_item_name in collect_list:
                        collected_item = np.array(data_point[collect_item_name])
                        # Append data to existing dataset
                        file[collect_item_name].resize((file[collect_item_name].shape[0] + collected_item.shape[0]), axis=0)
                        file[collect_item_name][-collected_item.shape[0]:] = collected_item

        print(f"Files loaded at:{post_processed_path}")



    def segment_study_(self,split:str,study_uid:str,series_uid:str,output_path:str):
        input_path      = os.path.join(self.dataset_path,split,study_uid,series_uid)
        output_path     = os.path.join(output_path,split,study_uid,series_uid)
        if directory_not_exists(output_path):
            totalsegmentator(input=input_path,output= output_path , preview=True)
            totalsegmentator(input=input_path,output= output_path , preview=True, task ="lung_vessels")
        else:
            print(f"Mask for {output_path} has alredy been created")
    
    def segment_studies(self,split:str,save_to:str,engine:Optional[str]="pyarrow",studies_colunm:Optional[str]="StudyInstanceUID",limits=None):
        """
        Load a study from the RSNA dataset.

        Parameters:
        - split (str): The split of the dataset.
        - df (pd.DataFrame): The DataFrame containing the data.
        - study_uid (str): The StudyInstanceUID of the study.
        - extension (str, optional): The file extension of DICOM files (default: '.dcm').

        Returns:
        - dict: A dictionary containing 'volume' (3D NumPy array) and 'labels' (study data as NumPy array).
        """


        print(f"Loading Model From: {os.getenv('TOTALSEG_WEIGHTS_PATH')}" )
        

        df                   = self.read_splits_(split=split,engine=engine) 
        unique_studies_uid   = self.get_unique_studies_(df=df,studies_colunm=studies_colunm) 
        

        if limits != None:
            if len(limits)  == 2:
                print(f"Running from limitis: {limits}")
                unique_studies_uid = unique_studies_uid[limits[0]: limits[1]]
            else:
                print("Limits must be a list fo upper and lowe bound")

        total_studies        = len(unique_studies_uid)

        
        for index, study_uid in tqdm(enumerate(unique_studies_uid), total=total_studies):
            series_uid = df[ df[studies_colunm] == study_uid]["SeriesInstanceUID"].to_list()[0]
            
            try:
                self.segment_study_(split=split,study_uid=study_uid,series_uid=series_uid ,output_path=save_to)
            except:
                print(f"Error at {series_uid }")


    def convert_study_to_nifti(self,df,split:str,study_uid:str,series_uid:str,output_path:str):
        input_path      = Path(os.path.join(self.dataset_path,split,study_uid,series_uid))
        output_path     = Path(os.path.join(output_path,split,study_uid,series_uid))

        
        
        #if directory_not_exists(output_path):
        #dcm_to_nifti(input_path,output_path)

        dicom_to_nii_gz(input_path, output_path )

        return output_path,split,study_uid,series_uid
        

        #dicom2nifti.dicom_series_to_nifti(input_path, output_path, reorient_nifti=True)

        #else:
         #   print(f"Mask for {output_path} has alredy been created")

    
    def collect_labels(self,df:pd.DataFrame,study_uid:str):
        study_df       = df[ df["StudyInstanceUID"] == study_uid]
    
        study_df.pop("StudyInstanceUID")
        study_df.pop("SeriesInstanceUID")
        study_df.pop("SOPInstanceUID")

        labels  = study_df.to_numpy()
        labels  = labels.astype(np.int8)
        data_point = {"labels":labels}


        return  data_point

    def convert_studies_to_nifti(self,split:str,save_to:str,engine:Optional[str]="pyarrow",studies_colunm:Optional[str]="StudyInstanceUID",limits=None):
        """
        Load a study from the RSNA dataset.

        Parameters:
        - split (str): The split of the dataset.
        - df (pd.DataFrame): The DataFrame containing the data.
        - study_uid (str): The StudyInstanceUID of the study.
        - extension (str, optional): The file extension of DICOM files (default: '.dcm').

        Returns:
        - dict: A dictionary containing 'volume' (3D NumPy array) and 'labels' (study data as NumPy array).
        """


        print(f"Loading Model From: {os.getenv('TOTALSEG_WEIGHTS_PATH')}" )
        
        df                   = self.read_splits_(split=split,engine=engine) 
        unique_studies_uid   = self.get_unique_studies_(df=df,studies_colunm=studies_colunm) 
        

        if limits != None:
            if len(limits)  == 2:
                print(f"Running from limitis: {limits}")
                unique_studies_uid = unique_studies_uid[limits[0]: limits[1]]
            else:
                print("Limits must be a list fo upper and lowe bound")

        total_studies        = len(unique_studies_uid)

        
        for index, study_uid in tqdm(enumerate(unique_studies_uid), total=total_studies):
            series_uid = df[ df[studies_colunm] == study_uid]["SeriesInstanceUID"].to_list()[0]
            
      
            output_path,split,study_uid,series_uid = self.convert_study_to_nifti(df=df,split=split,study_uid=study_uid,series_uid=series_uid ,output_path=save_to)
           
    

    def crop_studies_from_masks(self,split:str,nifti:str, masks:str,save_to:str,engine:Optional[str]="pyarrow",studies_colunm:Optional[str]="StudyInstanceUID",limits=None,
                        mask_roi_names= ["lung_lower_lobe_left.nii.gz",
                                         "lung_lower_lobe_right.nii.gz" ,
                                         "lung_middle_lobe_right.nii.gz" , 
                                         "lung_upper_lobe_left.nii.gz" ,
                                         "lung_upper_lobe_right.nii.gz" ,
                                         "pulmonary_artery.nii.gz"]):
        
        
        df                   = self.read_splits_(split=split,engine=engine) 
        unique_studies_uid   = self.get_unique_studies_(df=df,studies_colunm=studies_colunm) 
        

        if limits != None:
            if len(limits)  == 2:
                print(f"Running from limitis: {limits}")
                unique_studies_uid = unique_studies_uid[limits[0]: limits[1]]
            else:
                print("Limits must be a list fo upper and lowe bound")

        total_studies        = len(unique_studies_uid)

        error_count = 0
        for index, study_uid in tqdm(enumerate(unique_studies_uid), total=total_studies):
            series_uid = df[ df[studies_colunm] == study_uid]["SeriesInstanceUID"].to_list()[0]

            nifti_path    = Path(os.path.join(nifti,split,study_uid,series_uid))
            nifti_files   = self.find_nii_gz_files(nifti_path)
            output_path  = Path(os.path.join(save_to,split,study_uid,series_uid,"volume.nii.gz"))


            assert(len(nifti_files) == 1)
            nifti_path    = nifti_files[0]
            masks_path    = Path(os.path.join(masks,split,study_uid,series_uid))
           
            try:
                self._crop_study_from_mask(volume_path=nifti_path ,mask_roi_names=mask_roi_names,mask_path=masks_path,save_to= output_path  )
                #print("Volume saved")
            except: 
                print(f"Mask volume for {masks_path} does not exsists!")
                error_count += 1
            
            print(f"{error_count} Masks missing")
            
            
        
    @staticmethod
    def find_nii_gz_files(path):
        """
        Enumerate all files ending in '.nii.gz' in the given path.

        Args:
        path (str): The directory path to search for files.

        Returns:
        list: A list of the names of files ending in '.nii.gz'.
        """
        nii_gz_files = []
        for root, dirnames, filenames in os.walk(path):
            for filename in fnmatch.filter(filenames, '*.nii.gz'):
                nii_gz_files.append(os.path.join(root, filename))
        return nii_gz_files

    def _crop_study_from_mask(self,volume_path:str,mask_roi_names:list,mask_path:str,save_to:str):
        
        ### Msks ###
        mask_volume      = self.collect_volume_from_masks(mask_roi_names,mask_path)        # Collect volume from masks
        bbox             = get_bbox_from_mask(mask=mask_volume, outside_value=0, addon=0)  # Creat bounding box from masks selected
        
        ### Volumes ###
        voume_nib        = nib.load(volume_path)                    # Load volume                                    
        volume           = np.array(voume_nib.dataobj)              # convert nib to numpy
        
        ### Crop Volume
        croped_volume    = crop_to_bbox(image=volume, bbox=bbox  )                     # Crop using boundig box 
        croped_volume_nifti =  nib.Nifti1Image(croped_volume, affine=voume_nib.affine) # convert numpy to nifit 

        print("file")
        #import pdb;pdb.set_trace()
        save_to.parent.mkdir(parents=True, exist_ok=True)
        nib.save(croped_volume_nifti , save_to)
       
       
       
        
        
    
    @staticmethod
    def collect_volume_from_masks(mask_roi_names,mask_path):
        for i,mask_name in enumerate(mask_roi_names):
            mask_volume_nib =  nib.load(os.path.join(mask_path,mask_name))
            if i == 0:
                mask_volume     = np.array(mask_volume_nib .dataobj)
            else:
                mask_volume     += np.array(mask_volume_nib .dataobj)

        return mask_volume

        
 

        
        
    def simplfy_labels(self,split:str,save_to:str,engine:Optional[str]="pyarrow",studies_colunm:Optional[str]="StudyInstanceUID",limits=None):
        """
        Load a study from the RSNA dataset.

        Parameters:
        - split (str): The split of the dataset.
        - df (pd.DataFrame): The DataFrame containing the data.
        - study_uid (str): The StudyInstanceUID of the study.
        - extension (str, optional): The file extension of DICOM files (default: '.dcm').

        Returns:
        - dict: A dictionary containing 'volume' (3D NumPy array) and 'labels' (study data as NumPy array).
        """


        print(f"Loading Model From: {os.getenv('TOTALSEG_WEIGHTS_PATH')}" )
        
        df                   = self.read_splits_(split=split,engine=engine) 
        unique_studies_uid   = self.get_unique_studies_(df=df,studies_colunm=studies_colunm) 
     

        if limits != None:
            if len(limits)  == 2:
                print(f"Running from limitis: {limits}")
                unique_studies_uid = unique_studies_uid[limits[0]: limits[1]]
            else:
                print("Limits must be a list fo upper and lowe bound")

        total_studies        = len(unique_studies_uid)

        
        data_list = []
        limit_kloop = -1
        for index, study_uid in tqdm(enumerate(unique_studies_uid), total=total_studies):
            series_uid = df[df[studies_colunm] == study_uid]["SeriesInstanceUID"].to_list()[0]
            
            ### Label definition ###
            if df[df[studies_colunm] == study_uid]['pe_present_on_image'].sum() > 0:
                label = 1
            else:
                label = 0
            
            
            data_list.append([split,study_uid, series_uid, label]) # Append the data to the list

            if limit_kloop == index:
                import pdb; pdb.set_trace()
           

        # Create a DataFrame from the list
        df_result = pd.DataFrame(data_list, columns=['split', 'study_uid', 'series_uid', 'label'])

        # Save the DataFrame to a CSV file
        df_result.to_csv(os.path.join(save_to,f"{split}.csv"), index=False)
                    
                    
         
   
    


def dicom_to_nii_gz(dicom_directory, output_directory):
    # Convert DICOM to NIfTI
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    dicom2nifti.convert_directory(dicom_directory, output_directory)
