import torch.nn as nn
import torchvision



class ResNetUnoOcho(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.model = torchvision.models.resnet18(weights='DEFAULT')
        self.model.fc = nn.Linear(512,num_classes)
    
    def forward(self, x):
        x= self.model(x)
        return x
    
    def is_monogenic(self):
        return False