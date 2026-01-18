# FHRB-Net

## Introduction

This repo is the official implementation of ["Frequency-Domain Heterogeneous Rank-Entropy
Bipolarization Network for Remote Sensing Change Detection"]

## Install dependencies

1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)
2. [Install Pytorch 1.12 or later](https://pytorch.org/get-started/locally/)
3. Install dependencies

​	Use the following code in command line to install dependencies.

`	pip install -r requirements.txt`

## Data

Using any change detection dataset you want, but organize dataset path as follows. `dataset_name`  is name of change detection dataset, you can set whatever you want.

```python
dataset_name
├─train
│  ├─label
│  ├─t1
│  └─t2
├─val
│  ├─label
│  ├─t1
│  └─t2
└─test
    ├─label
    ├─t1
    └─t2
```

Below are some binary change detection dataset you may want.

[WHU Building](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html)

Paper: Fully convolutional networks for multisource building extraction from an open aerial and satellite imagery data set

[LEVIR-CD](https://justchenhao.github.io/LEVIR/)

Paper: A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection

[SYSU-CD](https://hub.fastgit.org/liumency/SYSU-CD)

Paper: SYSU-CD: A new change detection dataset in "A Deeply-supervised Attention Metric-based Network and an Open Aerial Image Dataset for Remote Sensing Change Detection"

[CDD](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9)

Paper: CHANGE DETECTION IN REMOTE SENSING IMAGES USING CONDITIONAL ADVERSARIAL NETWORKS


## Start

For training, run the following code in command line.

`python train.py`

If you want to debug while training, run the following code in command line.

`python -m ipdb train.py`

For test and inference, run the following code in command line.

`python inference.py` 

## Config

All the configs of dataset, training, validation and test are put in the file "utils/path_hyperparameter.py", you can change the configs in this file.

## Citation

If you use this work in your research, please cite:

