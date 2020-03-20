#!/usr/bin/env python
"""
Setting up PyTorch Lighting Trainer for CPU
and training a classication model using DenseNet.

Intended for development and simple models which are fast
"""
import argparse
import os
from pytorch_lightning import Trainer
from lightning_covid19.model.densenet import DenseNetModel
from lightning_covid19.utils import set_global_seeds

set_global_seeds(42)  # need to be before importing module


def get_args():
    # these are project-wide arguments
    parent_parser = argparse.ArgumentParser(__doc__, add_help=False)
    parent_parser.add_argument('--fast_dev_run',
                               default=False, action='store_true',
                               help='fast_dev_run: runs 1 batch of train, test, val (ie: a unit test)')
    parent_parser.add_argument('--num_gpus', default=0, type=int,
            help='See https://github.com/PyTorchLightning/pytorch-lightning/blob/732eaee4d735dafd1c90728e5583341d75ff72b5/pytorch_lightning/trainer/distrib_parts.py#L595')
    # add model specific arguments
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parser = DenseNetModel.add_model_specific_args(parent_parser, root_dir)

    return parser.parse_args()


def main():
    os.nice(10)  # Be nicer to your colleagues using shared CPU resources
    hparams = get_args()
    denseNet_model = DenseNetModel(hparams)
    trainer = Trainer(max_epochs=hparams.max_epochs,
                      fast_dev_run=hparams.fast_dev_run,
                      )
    trainer.fit(denseNet_model)
    trainer.test()


if __name__ == "__main__":
    main()
