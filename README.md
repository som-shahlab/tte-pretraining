# tte-pretraining
Code for the paper "Time-to-event pretraining for 3D medical imaging". You can **[read the paper here](https://arxiv.org/abs/2411.09361)**

We have provided the code for:

- [Installation](#installation)
- [Dataset](#dataset)
- [Tokenization](#tokenization)
- [Pretraining with parallel GPUs](#pretraining)
- [Evaluation with linear probe](#evaluation)
- [Tutorial for deriving tte training loss](#tutorial)


## Installation

You should install the required packages first

```bash
cd tte_pretraining
pip install -r docs/requirements.txt
conda install --file docs/environment.txt
```

## Dataset

You should direct to [here](https://aimi.stanford.edu/datasets/inspect-Multimodal-Dataset-for-Pulmonary-Embolism-Diagnosis-and-Prognosis) to download the image dataset in NIfTI format (i.e. file extensions as `.nii.gz`)
- The path to this folder will be used as `nii_folder` in below commands

You should direct to [here](https://redivis.com/datasets/dzc6-9jyt6gapt) to download EHR modality data in MEDS format (filename `meds_omop_inspect.tar.gz`)
- The path to this folder be be used as `parquet_folder` in below commands



## Tokenization

The tokenization process is to organize the EHR code into hierarchical form based on their ontology and then rank them based on entropy and other processing (e.g. normalizing given counts of patients with the code). Then eventually save the tokenizer

![Data Curation Process](tte_pretraining/docs/data_curation.png)

The number of pretraining tasks we select is 8,192 and the vocabulary size (total unique codes from EHR) is 65,535. You will need to download ontology from Athena. 

Athena is an OHDSI service for downloading ontologies. Simply visit https://athena.ohdsi.org, create an account, and click download at the top and put the ontology in the path in bash file.

Note: the downloaded ontology can be too large (i.e. few hundred GB) so optionally you want to prune it to fit to our dataset to make running substantially faster:

```bash
python 1a_prune_ontology.py \
--input-dataset "/share/pi/nigam/projects/zphuo/data/PE/inspect/timelines_smallfiles_meds/data/*parquet" \
--input-ontology "/share/pi/nigam/projects/zphuo/data/PE/inspect/ontology.pkl" \
--output-ontology "/share/pi/nigam/projects/zphuo/data/PE/inspect/inspect_ontology.pkl" \
--num-processors 32 
```

After that you can start training a tokenizer and save it:

```bash
cd tte_pretraining/training/
./1a_tokenizer.sh
```


## Pretraining

For pretraining we used 3 model architectures (SWINUNETR/ResNet/DenseNet)
- where SWINUNETR's pretrianing weights is from training on 50k public available CT/MRI dataset (weights can be download from [here to load in torch](https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt))
- ResNet and DenseNet are initialized from inflating 2D weights of pretrained data of ImageNet. The inflation process can be followed by [this instructions](https://github.com/hassony2/inflated_convnets_pytorch)
    - The script to conduct the operations are `tte_pretraining/training/src/i3dense.py`
    - And `tte_pretraining/training/src/i3res.py`

![Pretraining overview](tte_pretraining/docs/pretrain.png)

You can should specify the pretrained tokenizer from above and the dataset path (the `parquet` file folder) and image data path (`.nii.gz` files folder)

There are other hyperparameter training for the three architecture, you should refer to the [hyperparameter table](https://arxiv.org/pdf/2411.09361#page=21.10) for detailed reference when you input them into the bash script

```bash
cd tte_pretraining/training/
./1_pretrain_TTE_H100run_ddp.sh
```

Each of the architecture would require different training clocktime (or GPU time) with rough estimate.

| Architecture              | Number of GPUs        | Estimated wall-clock time | Estimated GPU hours |
|---------------------------|------------------------|----------------------------|----------------------|
| SwinUNETR<sub>base/TTE</sub>     | 4 H100 (80GB)           | 15 days                    | 1,440 GPU hours      |
| DenseNet-121<sub>base/TTE</sub>  | 4 A100 (40GB)           | 9 days                     | 864 GPU hours        |
| ResNet-152<sub>base/TTE</sub>    | 4 A100 (80GB)           | 10 days                    | 960 GPU hours        |


Note: optionally you can perform per task fine-tuning but this process is generally expensive given you need to train to completion for any downstream, i.e. `num_model * num_tasks` for full paremeter update and this tends not work well (per our [fine-tuning table results](https://arxiv.org/pdf/2411.09361#page=23.10)) but we also provide you script to to do fine-tuning as example

```bash
cd tte_pretraining/training/
./2_finetune_A100run_ddp.sh
```


## Evaluation

After pretraining is done we will perform linear probe (logistic regressin on binary classification tasks, and CoX-PH head of DeepSurv for TTE tasks).

![Task Adaptation](tte_pretraining/docs/linear_probe.png)

```bash
cd /share/pi/nigam/projects/zphuo/repos/tte-pretraining/tte_pretraining/training
./3_inference_TTE_H100_ddp.sh
```

We also test on the [RSPECT data](https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/data) for the out-of-distribution diagnosis task only evaluation

```bash
cd tte_pretraining/training
./3_inference_TTE_RSNA.sh
```



## Tutorial

We also provide guide for deriving tte training loss with exemplar CTs and their corresponding future codes as TTE tasks.

Please refer to notebook at `tutorial/pretrain_TTE_tutorial.ipynb`

Note: 
- This notebook doesn't require GPU to run but just CPU so the speed will slower but it only uses 1 CT as an example
- It still requires all the needed `nii_folder`, `parquet_folder`, `ontology_path` to derive TTE loss
- We reduced the `vocab_size` to 512 and `num_tasks` to 200 to improve speed of getting results
- The tutorial will prefit a bias term of the piecewise exponential model layer to avoid collapse without a good initial fit. This will take a few moments
- There's no gradient update or backpropagation, as we are only demonstrating deriving the loss term
