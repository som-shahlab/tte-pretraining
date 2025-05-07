import os
from dotenv import load_dotenv
from pathlib import Path
from totalsegmentator.python_api import totalsegmentator

# Load environment variables from .env file
load_dotenv()
# os.environ["TOTALSEG_WEIGHTS_PATH"] = os.getenv("TOTALSEG_WEIGHTS_PATH")
# print(f"Loading Model From: {os.getenv('TOTALSEG_WEIGHTS_PATH')}")

# # RSNA PE
# study = "4833c9b6a5d0"
# series = "57e3e3c5f910"
# split = "train"
# dicom_path = f"/datasets/RSNA_PE/{split}/{study}/{series}"
# output_path = dicom_path.replace(split, "rendered")

# INSPECT
dicom_path = "/datasets/PE/inspect/anon_dicoms_tar/2222-02-15T10:07:00"
output_path = dicom_path


seg = totalsegmentator(
    # input="//data/RSNAPE/test/ff62ec60c99b/0e5fa221590c",
    input=dicom_path,
    output=output_path,
    # roi_subset=["lung"],
    fast=True,
    # task="lung_vessels",
    preview=True,
)
print(seg.shape)
# seg = totalsegmentator(
#     input="/datasets/RSNAPE/RSNAPE/test/ff62ec60c99b/0e5fa221590c",
#     output="0e5fa221590c",
#     preview=True,
#     task="lung_vessels",
# )
# print(seg.shape)
