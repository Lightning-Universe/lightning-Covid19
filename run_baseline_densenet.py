#!/usr/bin/env python
"""
Setting up pytorch lighting Trainer for CPU
and training a classication model using DenseNet.

Intended for development and simple models which are fast
"""
import argparse
import os
import random
from pytorch_lightning import Trainer
from lightning_covid19.model.base_densenet import DenseNetModel
from lightning_covid19.utils import set_global_seeds

set_global_seeds(42)  # need to be before importing module
os.nice(10)  # Be nicer to your colleagues using shared CPU resources


# these are project-wide arguments
parent_parser = argparse.ArgumentParser(__doc__, add_help=False)
parent_parser.add_argument('--fast_dev_run', default=False, action='store_true',
   help='fast_dev_run: runs 1 batch of train, test  and val to find any bugs (ie: a unit test).')
# add model specific arguments
root_dir = os.path.dirname(os.path.realpath(__file__))
parser = DenseNetModel.add_model_specific_args(parent_parser, root_dir)

hparams = parser.parse_args()

denseNet_model = DenseNetModel(hparams)
trainer = Trainer(max_epochs=hparams.max_epochs,
                  fast_dev_run=hparams.fast_dev_run,
                  )
trainer.fit(denseNet_model)
trainer.test()
