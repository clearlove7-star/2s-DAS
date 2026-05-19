# 2s-DAS

[**论文链接 / 2s-DAS: Two-Stream Diffusion with Multi-Modal Fusion for Temporal Action Segmentation**](https://your_paper_link_here)

## Introduction
This project is an open-source implementation for **2s-DAS: Two-Stream Diffusion with Multi-Modal Fusion for Temporal Action Segmentation**, including full training code, inference scripts, and dataset. It aims to provide an efficient and 
reproducible research framework for temporal action segmentation.

## Environment Setup
- Python == 3.9
- PyTorch == 2.0.1
- Cuda == 11.8

## Dataset Download

The dataset is available at the links above.

Raw video files are needed to extract features. Please download the datasets with RGB videos from the official websites ([Breakfast](https://serre.lab.brown.edu/breakfast-actions-dataset.html) / [GTEA](https://cbs.ic.gatech.edu/fpv/) /[50Salads](https://cvip.computing.dundee.ac.uk/datasets/foodpreparation/50salads/)) and save them under the folder ./data/(name_dataset). 
  
## Extract Features
Extract features of 50salads, GTEA and Breakfast provided by [Br-Prompt](https://github.com/ttlmh/Bridge-Prompt) and [I3D](https://github.com/piergiaj/pytorch-i3d).

## Train your own model
you can retrain the model by yourself with following command.

Generate config files by ` python default_configs.py `

run ` python main_two_stream.py  --config configs/some_config.json  --device gpu_id `

Trained models and logs will be saved in the `result` folder

test ` python eval.py `

test one model ` python predict.py  --config configs/some_config.json  --device gpu_id `

Our model adapted form [DiffAct](https://github.com/Finspire13/DiffAct).

## Citation
