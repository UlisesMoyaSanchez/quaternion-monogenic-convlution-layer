#!/usr/bin/python
# -*- coding: utf-8 -*-


__author__ = ["E. Ulises Moya", "Abraham Sanchez"]
__copyright__ = "Copyright 2022, Gobierno de Jalisco, Universidad Autonoma de Guadalajara"
__credits__ = ["E. Ulises Moya", ]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = ["E. Ulises Moya", "Abraham Sanchez"]
__email__ = "eduardo.moya@jalisco.gob.mx"
__status__ = "Development"

import argparse, torch, os
from modules.module_cif10_res50_mon_lay import Netmonogenic, time
from pytorch_lightning import loggers, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
# from torchsummary import summary
from pytorch_lightning.loggers import CSVLogger
from modules.back_ends import ResNetBackEnd, Bottleneck
from layers.monogenic import Monogenic
from utils import Utils

if __name__ == '__main__':
    train_model_on = True
    plot_images_on = False
    monogenic_net_on = True
    quality_img_on = False
    fourier_on = False
    mon = Monogenic()
    utils = Utils()

    parser = argparse.ArgumentParser()
    parser = Netmonogenic.add_model_specific_args(parent_parser=parser)
    parser = Trainer.add_argparse_args(parent_parser=parser)
    args = parser.parse_args()

    work_dir, init_datetime, init_time = os.path.abspath(os.curdir), utils.getDateTime(), time.time()
    logger_path = f'{args.general_storage}/{args.id_name}'
    cvs_logger = CSVLogger(logger_path)
    torch.set_float32_matmul_precision('medium') if args.fp == 16 else torch.set_float32_matmul_precision('high')

    checkpoint = ModelCheckpoint(
        monitor='val_avg_acc', mode='max', verbose=True, dirpath=args.general_storage+'/'+args.id_name,
        filename='test-{epoch:03d}')

    trainer = Trainer.from_argparse_args(args, callbacks=checkpoint, logger=[cvs_logger],
                                         enable_progress_bar=False, check_val_every_n_epoch=1, precision=args.fp,
                                         log_every_n_steps=250, accelerator=args.accelerator, devices=args.devices)

    model = Netmonogenic(args)

    if train_model_on:

        model.plot_images = plot_images_on                      ### Activate plots storage
        model.monogenic_net = monogenic_net_on                  ### Activate Monogenic layer within model

        model.m_train = False                                   ### Activate plots storage for training
        trainer.fit(model=model)
        model.m_train = False                                   ### Deactivate plots storage for training

        ckpt_path = checkpoint.best_model_path
        print(f'\nBest checkpoint path: {ckpt_path}')

        ### ------------ save best epoch data in CSV ------------------------------------------
        checkpoint = torch.load(ckpt_path)                  ### Load checkpoint from pretained model
        k, v = list(checkpoint['callbacks'].items())[0]     ### Get keys and values from dictionary of checkpoint
        ### Define parameters to store on summary CSV file
        model_used = f'FP-{args.fp} Net_with_Monogenic' if monogenic_net_on else f'FP-{args.fp} Net_without_Monogenic'
        utils.store_results(args.general_storage + '/cifar10_mon_performance.csv',
                            [f'Training_{model_used}', init_datetime, utils.getDateTime(),
                             f'{(time.time() - init_time)/60:.2f}', args.max_epochs, args.batch_size, args.learning_rate,
                             args.momentum,
                             args.weight_decay, checkpoint['epoch'], f'{checkpoint["callbacks"][k]["best_model_score"].item():.4f}',
                             '-', f'{mon.sigmax.item()}_'
                                  f'{mon.sigmay.item()}_'
                                  f'{mon.wave_length1.item()}',
                             args.workers, checkpoint['callbacks'][k]['best_model_path'],
                             f'{logger_path}/lightning_logs/version_{cvs_logger.version}'])

        ### Print best epoch data obtained from training
        print(f'\nTrain Ended | Epoch max. acc. {checkpoint["epoch"]} | loss {"-"} | '
              f'acc: {checkpoint["callbacks"][k]["best_model_score"].item():.4f} | '
              f'time: {(time.time() - init_time)/60:.2f} min. | '
              f'best_sx: {mon.sigmax.item()} | '
              f'best_sy: {mon.sigmay.item()} | '
              f'best_wl: {mon.wave_length1.item()}')

    ### --------------------------------- Test model with checkpoint ..................................
    import Test_model_mon_lay, utils, numpy as np

    epsilon = [[0.5]]               ### Epsilon for attack
    for e in epsilon:
        epsilon = e
        if not train_model_on:
            ckpt_path = f'{args.general_storage}/os/test-epoch=000-v11.ckpt'

        test = Test_model_mon_lay.TestModel(args, ckpt_path)                       ### Get Class to obtain objects
        t_e = test.test(plot_images_on, monogenic_net_on)                                           ### Test images with pretrained model
        a_t_e = test.test_attack(epsilon, plot_images_on, monogenic_net_on, steps=1)               ### Generate adversarial examples and test with pretrained model

        test.image_quality_measure(t_e, a_t_e, monogenic_net_on, plot_images_on, quality_img_on, fourier_on)
