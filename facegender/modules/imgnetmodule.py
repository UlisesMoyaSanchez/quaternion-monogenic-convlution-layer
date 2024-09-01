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

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.functional import cross_entropy
#from lightning.pytorch  import LightningModule
from torchmetrics import Accuracy
from pytorch_lightning import LightningModule
from modules.factory import MonoFactory


class Netmonogenic(LightningModule):
    
    def __init__(self, learning_rate, momentum, weight_decay,mtype, num_classes):
        super(Netmonogenic, self).__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.val_acc = Accuracy(top_k=1,task='multiclass',num_classes=num_classes)
        self.train_acc = Accuracy(top_k=1, task='multiclass',num_classes=num_classes)
        self.train_loss = list()
        self.val_loss = list()
        self.test_loss = list()
        self.test_acc = Accuracy(top_k=1, task='multiclass',num_classes=num_classes)
        self.model = MonoFactory.create(mtype=mtype,num_classes=num_classes)
        
 
    
    def forward(self, x):
        x = self.model(x)
        return x 


    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        #optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-5)
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 10))
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        preds = self(images)
        loss = cross_entropy(preds, targets)
        acc = self.train_acc(preds, targets)
        if self.model.is_monogenic():
            mono = self.model.get_mono_weigths()
            sx1, sy1, wl1,  sx2, sy2, wl2, sx3, sy3, wl3 = mono.parameters()
            metrics = {
                'train_loss': loss.detach(),
                'train_acc': acc,
                'sx1': sx1.detach(),
                'sy1': sy1.detach(),
                'wl1': wl1.detach(),
                'sx2': sx2.detach(),
                'sy2': sy2.detach(),
                'wl2': wl2.detach(),
                'sx3': sx3.detach(),
                'sy3': sy3.detach(),
                'wl3': wl3.detach(),
            }
            self.log_dict(metrics, prog_bar=True)
        self.train_loss.append(loss.detach())
        return {'loss': loss}
        #return {'loss': loss, 'train_loss': loss.detach(), 'train_acc': acc, 'sx': sx.detach(), 'sy':sy.detach(), 'wl': wl.detach()}

        
    
    def on_train_epoch_end(self):
        mean_loss = torch.stack(self.train_loss).mean()
        mean_acc = self.train_acc.compute()
        metrics = {'train_avg_loss': mean_loss, 'train_avg_acc': mean_acc}
        self.log_dict(metrics, prog_bar=True)
        self.train_loss.clear()


    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self(images)
        loss = cross_entropy(preds, targets)
        acc = self.val_acc(preds, targets)
        metrics = {
            'val_loss': loss.detach(),
            'val_acc': acc
        }
        self.log_dict(metrics)
        self.val_loss.append(loss.detach())
        return {'loss': loss}
        #return {'loss': loss, 'val_loss': loss.detach(), 'val_acc': acc}

    

    def on_validation_epoch_end(self):
        mean_loss = torch.stack(self.val_loss).mean()
        mean_acc = self.val_acc.compute()
        metrics = {'val_avg_loss': mean_loss, 'val_avg_acc': mean_acc}
        self.log_dict(metrics, prog_bar=True)
        self.val_loss.clear()
    
    def test_step(self, batch, batch_idx):
        images, targets = batch
        preds = self(images)
        loss = cross_entropy(preds, targets)
        acc = self.test_acc(preds, targets)
        metrics = {
            'test_loss': loss.detach(),
            'test_acc': acc
        }
        self.log_dict(metrics)
        self.test_loss.append(loss.detach())
        return {'loss': loss}
        #return {'loss': loss, 'val_loss': loss.detach(), 'val_acc': acc}

    

    def on_test_epoch_end(self):
        mean_loss = torch.stack(self.test_loss).mean()
        mean_acc = self.test_acc.compute()
        metrics = {'test_avg_loss': mean_loss, 'test_avg_acc': mean_acc}
        self.log_dict(metrics, prog_bar=True)
        self.test_loss.clear()
