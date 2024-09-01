#!/usr/bin/python
# -*- coding: utf-8 -*-


__author__ = ["E. Ulises Moya", "Abraham Sanchez"]
__copyright__ = "Copyright 2022, Eduardo Ulises Moya Sanchez"
__credits__ = ["E. Ulises Moya", ]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = ["E. Ulises Moya", "Abraham Sanchez"]
__email__ = "eduardo.moya@jalisco.gob.mx"
__status__ = "Development"



#from lightning.pytorch.utilities.cli import LightningCLI
# runs
from pytorch_lightning.cli import LightningCLI
#from lightning.pytorch.cli import LightningCLI
from modules.imgnetmodule import Netmonogenic
from modules.data_monogenic import DataMonogenic

def main() -> None:
    LightningCLI(model_class=Netmonogenic,
                 datamodule_class=DataMonogenic,
                 save_config_kwargs={"overwrite": True})

if __name__ == '__main__':
    print("---3 channels---")
    main()
