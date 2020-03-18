# Lightning-Covid19

[![Build Status](https://travis-ci.org/PyTorchLightning/lightning-Covid19.svg?branch=master)](https://travis-ci.org/PyTorchLightning/lightning-Covid19)

A detector for [covid-19 chest X-ray images](https://github.com/ieee8023/covid-chestxray-dataset) 
 using [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) (for educational purposes)

## Overview

This project came with actual global situation and reciently release X-ray dataset.

## Data

The baseline models uses only [covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset/)
 which is rather small (79 images at the time of writing).

They are also accessible with following python packages:
- https://github.com/ieee8023/covid-chestxray-dataset
- https://github.com/mlmed/torchxrayvision

## Installation

  python3.6 -m venv venv  # from repo root, use python3.6+
  source venv/bin/activate
  pip install -r requirements

## Contribution

Anyone is welcome to contribute and use this project...
For easier combination and coordination we are using separate [channel](https://pytorch-lightning.slack.com/archives/CV7MNM0NP) in Lightning Slack, feel free to join!

## References

TBD

## The baseline experiment
See 
```bash
  ./lightning_covid19/model/base_densenet.py:DenseNetModel
  ./run_baseline_densenet.py
```
Run 
```bash
  ./run_baseline_densenet.py
```

### Model

The `DenseNet` model is taken from [torchxrayvision library](https://github.com/mlmed/torchxrayvision) 

### Experiment setup

- Single node, SGD, GPU, single node, training time 12min on 4 CPU workers
- Metrics: `val_loss`, `accuracy`
- Results: `val_acc` and `val_loss` highly unstable -- too small data
  - The model starts over fitting somewhere after 13th epoch.

```
Epoch 4: : 50it [00:01,  4.05it/s, loss=0.625, train_loss=0.294, v_num=14, val_acc=0.75, val_loss=0.344]
Epoch 7: : 50it [00:01,  3.73it/s, loss=0.503, train_loss=0.377, v_num=14, val_acc=0.917, val_loss=0.33]
Epoch 8: : 50it [00:01,  3.72it/s, loss=0.465, train_loss=0.212, v_num=14, val_acc=0.667, val_loss=0.43]
Epoch 8: : 50it [00:17,  3.72it/s, loss=0.465, train_loss=0.212, v_num=14, val_acc=0.667, val_loss=0.43]
Epoch 9: : 50it [00:04,  3.24it/s, loss=0.430, train_loss=0.303, v_num=14, val_acc=1, val_loss=0.243]
Epoch 10: : 50it [00:02,  3.08it/s, loss=0.408, train_loss=0.27, v_num=14, val_acc=1, val_loss=0.198]
Epoch 11: : 50it [00:21,  3.30it/s, loss=0.373, train_loss=0.177, v_num=14, val_acc=1, val_loss=0.177]
Epoch 12: : 50it [00:01,  3.37it/s, loss=0.366, train_loss=0.252, v_num=14, val_acc=1, val_loss=0.213]
Epoch 13: : 50it [00:02,  3.31it/s, loss=0.357, train_loss=0.444, v_num=14, val_acc=1, val_loss=0.172]
Epoch 16: : 50it [00:02,  3.45it/s, loss=0.313, train_loss=0.319, v_num=14, val_acc=0.667, val_loss=0.362]
Epoch 17: : 50it [00:02,  3.37it/s, loss=0.310, train_loss=0.131, v_num=14, val_acc=0.667, val_loss=0.392]
Epoch 18: : 50it [00:01,  3.45it/s, loss=0.324, train_loss=0.689, v_num=14, val_acc=0.667, val_loss=0.394]
Epoch 19: : 50it [00:20,  3.61it/s, loss=0.338, train_loss=0.406, v_num=14, val_acc=1, val_loss=0.245]
Epoch 20: : 50it [00:01,  3.69it/s, loss=0.375, train_loss=0.167, v_num=14, val_acc=0.917, val_loss=0.236]
Epoch 20: : 50it [00:17,  3.69it/s, loss=0.375, train_loss=0.167, v_num=14, val_acc=0.917, val_loss=0.236]
```
