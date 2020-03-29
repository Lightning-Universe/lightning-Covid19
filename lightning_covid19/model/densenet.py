"""
Training a DenseNet model [1] on a covid-chestxray-dataset[2]
which is randomly splitted to training, validation, test sets (ratios 60/20/20).

[1] https://github.com/mlmed/torchxrayvision)
[2] https://github.com/ieee8023/covid-chestxray-dataset/
"""
import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import torchxrayvision as xrv

from pytorch_lightning.core import LightningModule
from lightning_covid19.utils import run

import lightning_covid19.model.augmentation as transforms


class DenseNetModel(LightningModule):
    """Training a DenseNet model [1] on a covid-chestxray-dataset[2]"""

    def __init__(self, hparams):
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param hparams:
        """
        super(DenseNetModel, self).__init__()
        self.hparams = hparams

        # TODO make it work
        # w = h = self.hparams.xraysize
        # num_channels = 1
        # if you specify an example input, the summary will show input/output for each layer
        # self.example_input_array = torch.rand(self.hparams.batch_size, num_channels, w, h)

        self.dense_net = xrv.models.DenseNet(num_classes=2)
        self.criterion = nn.CrossEntropyLoss()

        self.transform = transforms.DataAugmentator()

    def forward(self, x):
        logits = self.dense_net(x)
        return logits

    def _parse_batch(self, batch):
        x = batch['PA']
        x = self.transform(x)  # augment data at batch level

        y = batch['lab'].long()[:, 2]
        return x, y

    def training_step(self, batch, batch_idx):
        # forward pass
        x, y = self._parse_batch(batch)
        y_hat = self.forward(x)

        loss = self.criterion(y_hat, y)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        tqdm_dict = {'train_loss': loss}
        return {'loss': loss, 'progress_bar': tqdm_dict, 'log': tqdm_dict}

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, 'test')

    def _eval_step(self, batch, batch_idx, prefix):
        x, y = self._parse_batch(batch)
        y_hat = self.forward(x)

        loss = self.criterion(y_hat, y)

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        acc = torch.tensor(acc)

        if self.on_gpu:
            acc = acc.cuda(loss.device.index)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)
            acc = acc.unsqueeze(0)

        return {f'{prefix}_loss': loss, f'{prefix}_acc': acc}

    def validation_epoch_end(self, outputs):
        return self._eval_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        return self._eval_epoch_end(outputs, 'test')

    def _eval_epoch_end(self, outputs, prefix):
        """
        Called at the end of test/validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

        loss_mean = 0
        acc_mean = 0
        for output in outputs:
            loss = output[f'{prefix}_loss']

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                loss = torch.mean(loss)
            loss_mean += loss

            # reduce manually when using dp
            acc = output[f'{prefix}_acc']
            if self.trainer.use_dp or self.trainer.use_ddp2:
                acc = torch.mean(acc)

            acc_mean += acc

        loss_mean /= len(outputs)
        acc_mean /= len(outputs)
        tqdm_dict = {f'{prefix}_loss': loss_mean, f'{prefix}_acc': acc_mean}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, f'{prefix}_loss': loss_mean}
        return result

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    # ------------- Model hyperparameters with default values ------------- #

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        """
        Parameters you define here are available to your model through self.hparams
        :param parent_parser: parent_parser is set up typically in the main training script
        :param root_dir:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser])

        # data
        git_root = run('git rev-parse --show-toplevel').strip()
        parser.add_argument('--data_root', default=f'{git_root}/data/covid-chestxray-dataset', type=str)

        # nn & training params
        parser.add_argument('--max_epochs', default=20, type=int)
        parser.add_argument('--batch_size', default=4, type=int)
        parser.add_argument('--learning_rate', default=0.001, type=float)
        parser.add_argument('--xraysize', default=224, type=int, help='Do no change unless you change covid dataset')
        return parser

    # --------------------- Data preparation --------------------- #

    def prepare_data(self):
        chestxray_root = self.hparams.data_root
        clone_uri = 'https://github.com/ieee8023/covid-chestxray-dataset.git'
        if os.path.exists(chestxray_root):
            assert os.path.isdir(chestxray_root), f'{chestxray_root} should be cloned from {clone_uri}'
        else:
            print('Cloning the covid chestxray dataset. It may take a while\n...\n', flush=True)
            run(f'git clone {clone_uri} {chestxray_root}')

        transform = nn.Sequential(
            transforms.ToTensor(),
            transforms.XRayCenterCrop(),
            transforms.XRayResizer(self.hparams.xraysize),
            transforms.Squeeze(),
        )

        covid19 = xrv.datasets.COVID19_Dataset(
            imgpath=f'{chestxray_root}/images',
            csvpath=f'{chestxray_root}/metadata.csv',
            transform=transform)
        print(f'Covid Chest x-ray stats dataset stats:\n{covid19}\n\n', flush=True)

        # count split sizes
        n_train = int(0.8 * len(covid19))
        n_valid_test = len(covid19) - n_train
        n_valid = int(0.5 * n_valid_test)
        n_test = n_valid_test - n_valid

        # split the dataset
        self.train_set, self.val_set, self.test_set = \
            torch.utils.data.random_split(covid19, [n_train, n_valid, n_test])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.hparams.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.hparams.batch_size, shuffle=False)
