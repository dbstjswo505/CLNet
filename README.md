# Causal Localization Network for Radar Human Localization with micro-Doppler signature

## Environment
- python 3.7
- torch 1.13.1
- CUDA 12.2

## Setting
```
conda env create --file environment.yaml
```

## Download IDRad data
You can download IDRad dataset below two links

- [**IDRad_original**](https://www.imec-int.com/en/IDRad)

- [**IDRad**](https://drive.google.com/file/d/1xg6vjABcuhJu8rVPt9RNMiiYVEYDUHHm/view?usp=drive_link)

Option: download IDRad-TBA (Temporal Boundary Annotation) dataset below link

- [**IDRad-TBA**](https://drive.google.com/file/d/1MY8ikGtRSJQ05EB28YYpcTXXAWviFYtd/view?usp=sharing)

## Reproduce IDRad-TBA dataset for Radar Human Localization
```
python make_data.py
```
Prepare 'IDRad' folder for IDRad dataset

## Prepare causal mask and spasity mask for Causal Localization Network (CLNet)
```
python make_mask.py
```
The outputs are 'causal_mask.npy' and 'sparsity_mask.npy'

## Train CLNet for radar human localization task
```
python train.py
```
Inference only is also possible by commenting the training in the code.

## Reference
Dual-scale Doppler Attention for Human Identification. [Link](https://github.com/dbstjswo505/DSDA)

Indoor Person Identification with Radar Data. [Link](https://github.com/baptist/idrad)
