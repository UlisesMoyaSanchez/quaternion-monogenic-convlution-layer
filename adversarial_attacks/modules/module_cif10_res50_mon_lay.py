#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = ["E. Ulises Moya", "Abraham Sanchez"]
__copyright__ = "Copyright 2021, Gobierno de Jalisco, Universidad Autonoma de Guadalajara"
__credits__ = ["E. Ulises Moya", ]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = ["E. Ulises Moya", "Abraham Sanchez"]
__email__ = "eduardo.moya@jalisco.gob.mx"
__status__ = "Development"

import os, argparse, gc, torch, torchvision, torchvision.transforms as transforms
import torch.optim as optim, torch.optim.lr_scheduler as lr_scheduler, numpy as np, time
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from pytorch_lightning.core import LightningModule
from torchmetrics import Accuracy
from layers.monogenic import Monogenic
from .back_ends import ResNetBackEnd, Bottleneck

class Netmonogenic(LightningModule):

    def __init__(self, hparams, dnn_model=ResNetBackEnd(block=Bottleneck, layers=[3, 4, 6, 3], inplanes=6)):
        super(Netmonogenic, self).__init__()
        self.save_hyperparameters(hparams)
        self.val_acc = Accuracy(top_k=1, task='multiclass', num_classes=10)
        self.train_acc = Accuracy(top_k=1, task='multiclass', num_classes=10)
        self.test_acc = Accuracy(top_k=1, task='multiclass', num_classes=10)
        self.model = dnn_model

        ### ------------- new --------------------------------------------
        self.trans_mean = (0.4914, 0.4822, 0.4465)
        self.trans_std = (0.2023, 0.1994, 0.2010)
        self.work_dir = os.path.abspath(os.curdir)
        self.tic_epoch, self.init_tic = time.time(), time.time()
        self.m_train = False
        self.plot_images = False
        self.monogenic_net = True
        self.initial_lr = self.hparams.learning_rate
        self.new_lr = self.hparams.learning_rate
        self.plot_step = 100
        self.data_path = self.hparams.data_path
        ### ------------------------------------------------------------

    def forward(self, x):

        monogenic = Monogenic()
        ### ------------- new --------------------------------------------
        step_trigger = False
        if self.plot_images:
            if self.m_train:
                if self.len_train != 0 and self.plot_step > 0 and int(
                        self.global_step % (self.len_train / self.hparams.batch_size)) == self.plot_step:
                    step_trigger = True
        ### -------------------------------------------------------------

        if self.monogenic_net:
            monogenic.step_trigger = step_trigger
            x = monogenic.forward(x)

        x = self.model(x)

        return x

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.new_lr, momentum=self.hparams.momentum,
                              weight_decay=self.hparams.weight_decay)
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 10))

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):

        images, targets = batch

        monogenic = Monogenic()
        preds = self.forward(images)

        loss = cross_entropy(preds, targets)
        acc = self.train_acc(preds, targets)
        mono = monogenic
        sx, sy, wl = mono.sigmax, mono.sigmay, mono.wave_length1  # mono.parameters()
        metrics = {'train_loss': loss.detach(), 'train_acc': acc, 'sx': sx.item(), 'sy': sy.item(), 'wl': wl.item()}
        self.log_dict(metrics, prog_bar=True)

        return {'loss': loss, 'train_loss': loss.detach(), 'train_acc': acc, 'sx': sx.item(), 'sy': sy.item(),
                'wl': wl.item()}

    def training_epoch_end(self, outputs):
        monogenic = Monogenic()
        avg_loss = torch.stack([output['train_loss'] for output in outputs]).mean()
        avg_acc1 = torch.stack([output['train_acc'] for output in outputs]).mean()
        avg_sx = np.array([output['sx'] for output in outputs]).mean()
        avg_sy = np.array([output['sy'] for output in outputs]).mean()
        avg_wl = np.array([output['wl'] for output in outputs]).mean()

        metrics = {'train_avg_loss': avg_loss, 'train_avg_acc': avg_acc1, 'train_avg_sx': avg_sx,
                   'train_avg_sy': avg_sy, 'train_avg_wl': avg_wl}

        self.log_dict(metrics)
        gc.collect()

        ### ------------- print epoch end parameters status ---------------
        mono = monogenic
        sx, sy, wl = mono.sigmax, mono.sigmay, mono.wave_length1  #mono.parameters()
        print(f'Train - Epoch {self.current_epoch} | Avg. loss {avg_loss:.4f} | Avg. acc: {avg_acc1:.4f} | '
              f'time: {(time.time() - self.tic_epoch) / 60:.2f} min. | '
              f'avg_sx: {avg_sx:.3f} | avg_sy: {avg_sy:.3f} | avg_wl: {avg_wl:.3f}'
              f' | Last mon parameters: {sx:.5f} {sy:.5f} {wl:.5f}')
        self.tic_epoch = time.time()
        self.log_dir = str(self.logger.log_dir)
        ### --------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.forward(images)

        loss = cross_entropy(preds, targets)
        acc = self.val_acc(preds, targets)
        metrics = {'val_loss': loss.detach(), 'val_acc': acc}
        self.log_dict(metrics)

        return {'loss': loss, 'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        avg_acc1 = torch.stack([output['val_acc'] for output in outputs]).mean()
        metrics = {
            'val_avg_loss': avg_loss,
            'val_avg_acc': avg_acc1
        }
        self.log_dict(metrics)
        gc.collect()

        ### ------------- new --------------------------------------------
        if self.current_epoch >= 0:
            print(f'\nVal - Epoch {self.current_epoch} | Loss: {avg_loss:.3f} | Acc: {avg_acc1:.3f}')
        ### --------------------------------------------------------------

    def train_dataloader(self):
        monogenic = Monogenic()
        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             torchvision.transforms.RandomHorizontalFlip(),
             torchvision.transforms.ToTensor(),
             transforms.Normalize(self.trans_mean, self.trans_std)])

        train_set = torchvision.datasets.CIFAR10(root=str(self.data_path), train=True,
                                                 download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.hparams.batch_size,
                                                   shuffle=True, num_workers=self.hparams.workers, pin_memory=True)
        self.len_train = len(train_loader) * self.hparams.batch_size

        ### ------------- new --------------------------------------------
        print(f'\nLen train: {self.len_train} | Work dir: {self.work_dir} | FP-{self.hparams.fp} '
              f'| sx, sy, wl: {monogenic.sigmax.item(), monogenic.sigmay.item(), monogenic.wave_length1.item()}')
        ### --------------------------------------------------------------

        return train_loader

    def val_dataloader(self):
        transform_test = transforms.Compose(
            [torchvision.transforms.ToTensor(),
             transforms.Normalize(self.trans_mean, self.trans_std)])

        test_set = torchvision.datasets.CIFAR10(root=str(self.data_path), train=False,
                                                download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.hparams.batch_size,
                                                  shuffle=False, num_workers=self.hparams.workers, pin_memory=True)
        self.len_val = len(test_loader) * self.hparams.batch_size

        return test_loader

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('-l', '--learning_rate', type=float, default=0.001, required=False)
        parser.add_argument('-mom', '--momentum', type=float, default=0.9, required=False)
        parser.add_argument('-wd', '--weight_decay', type=float, default=1e-4, required=False)
        parser.add_argument('-w', '--workers', type=int, default=1, required=False)
        parser.add_argument('-b', '--batch_size', type=int, default=8, required=False)
        parser.add_argument('-in', '--input_size', type=int, default=224, required=False)
        parser.add_argument('-d', '--data_path', type=str, default='./data', required=False)
        parser.add_argument('-id', '--id_name', type=str, default='default', required=True)

        ### ----------------------------------- New -----------------------------------------
        parser.add_argument('-fp', '--fp', type=int, default=32, required=False)
        parser.add_argument('-stor', '--general_storage', type=str, default='./mon_storage',
                            required=False)
        ### ---------------------------------------------------------------------------------

        return parser
