#from lightning.pytorch import LightningDataModule
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os

class DataMonogenic(LightningDataModule):
    def __init__(
            self,
            path: str,
            batch_size: int,
            num_workers: int,
            input_size: int,
        ) -> None:
        super(DataMonogenic, self).__init__()
        self.data_path = path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = input_size

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
    
    def test_dataloader(self):
        return self.val_loader

    def setup(self,stage):
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15), 
            transforms.ToTensor()
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        #dataset  = torchvision.datasets.CIFAR10(root='./CIFAR10_data/', train=True, download=True, transform=transform)
        traindir = os.path.join(self.data_path, 'train')
        train_dir = ImageFolder(root = traindir, transform=train_transform)
        self.train_loader = DataLoader(train_dir, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        

        val_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.input_size),
            #transforms.Resize(256),
            #transforms.CenterCrop(self.hparams.input_size),
            transforms.ToTensor()
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])        
        valdir = os.path.join(self.data_path, "val")
        val_dir = ImageFolder(root = valdir, transform=val_transform)
        self.val_loader = DataLoader(val_dir, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)


        test_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.input_size),
            #transforms.Resize(256),
            #transforms.CenterCrop(self.hparams.input_size),
            transforms.ToTensor()
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])        
        testdir = os.path.join(self.data_path, "test")
        test_dir = ImageFolder(root = testdir, transform=test_transform)
        self.test_loader = DataLoader(test_dir, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
