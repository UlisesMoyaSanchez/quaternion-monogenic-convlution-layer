import torch.nn as nn
import torchvision
import torch
from layers.monogenic import Monogenic



class MonoResNet18(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.model = torchvision.models.resnet18(weights='DEFAULT')
        self.mono = Monogenic()
        self.add_channels()
        self.model.fc = nn.Linear(512,num_classes)
 


    def add_channels(self):
        conv1 = self.model.conv1.weight.clone()
        conv1_w_idx0 = self.model.conv1.weight[:, 0].clone()
        self.model.conv1 = nn.Conv2d(18,64,7,2,3,bias=False)
        with torch.no_grad():
            self.model.conv1.weight[:,:3] = conv1
            for idx in range(3, 18):
                self.model.conv1.weight[:, idx].weight = conv1_w_idx0
    
    def forward(self, x):
        x=self.mono(x)
        x= self.model(x)
        return x
    
    def get_mono_weigths(self):
        return self.mono
    
    def is_monogenic(self):
        return True
        



