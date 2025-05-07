# Unit Test for 3D pretrained image model for training/inference

### out-of-the-box inference and adaptation 

We pretrained a DenseNet model with TTE objective under Stanford data (INSPECT cohort). We aim to use this code base as example for how to use it

User need to supply input data (e.g. 3D CT scans in nii.gz format) and label data (binary labels)

The model weights can be downloaded from: https://huggingface.co/StanfordShahLab

It should be put to `data_temp` folder

Then you can run below to get example results:

```python test3D.py``` 



Note: the program's functionality is:
- It uses a public dataset for demo so anyone can run it
- It loads a model weight (you need to download from above if you have stanford email)
- Then user needs to supply labels so that the embeddings can eventually be mapped to it
- It trains a logistic regression given frozen model, and eval
- Optionally the model can be fully fine tuned then do the eval again


### Unit test:

1) model weights should be loaded properly. We will load the ones that we need from TTE-pretrained model

2) The features/labels should match as expected. The TTE pretrained DenseNet is trained with 1024 dim

Example to run the test:
```
cd tte-pretraining/tests 
pytest image_pytest.py 
```



