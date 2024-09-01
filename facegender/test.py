import torchvision
import torch.nn as nn
import torch
from layers.monogenic import Monogenic

model = torchvision.models.resnet18(weights='DEFAULT')

mono = Monogenic()
mono.cuda()

conv1 = model.conv1.weight.clone()
conv1_w_idx0 = model.conv1.weight[:, 0].clone()
model.conv1 = nn.Conv2d(18,64,7,2,3,bias=False)
model.fc = nn.Linear(512,2)
with torch.no_grad():
    model.conv1.weight[:,:3] = conv1
    for idx in range(3, 18):
        model.conv1.weight[:, idx].weight = conv1_w_idx0

model.cuda()
tensor = torch.ones((1,3,90,90)).cuda()

print(model)

x = mono(tensor)
x = model(x)

