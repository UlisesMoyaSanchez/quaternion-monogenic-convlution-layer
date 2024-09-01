from modules.monoresnet18 import MonoResNet18
from modules.resnet_unoocho import ResNetUnoOcho

class MonoFactory:
    @staticmethod
    def create(mtype,num_classes):
        if mtype == 'monoresnet18':
            return MonoResNet18(num_classes)
        elif mtype == 'resnet18':
            return ResNetUnoOcho(num_classes)


