import torch
from models.options import ParamOptions
from models.trainer import TrainModel
from utils2.visualizer import Visualizer
import numpy as np
import os
import time
from models.lossplot1 import plotmetrics

from torch import nn


defaultkwargs = {
    'run_path': F'{os.getcwd()}/results',
    'run_name': time.strftime("%y_%m_%d-%H_%M"),
    'num_samples': 10000,                                                       # number of randomly sampled image patches
    'batch_size': 4,
    'data_A': 'dataset/T700_GF_t05_pxs08.json',
    'data_B': 'dataset/T700_pco_pxs04.json',
    'lossFuncs_G_A': [nn.BCELoss()],
    'lossNames_G_A': ['G_A_BCE'],
    'lambdas_G_A': [10],
    'lossChannels_G_A': None,
    'lossFuncs_G_B': [nn.BCELoss()],
    'lossNames_G_B': ['G_B_BCE'],
    'lambdas_G_B': [10],
    'lossChannels_G_B': None,
    'lossFuncs_D_A': [nn.BCELoss()],
    'lossNames_D_A': ['D_A_BCE'],
    'lambdas_D_A': [1],
    'lossChannels_D_A': None,
    'lossFuncs_D_B': [nn.BCELoss()],
    'lossNames_D_B': ['D_B_BCE'],
    'lambdas_D_B': [1],
    'lossChannels_D_B': None,
    'lossFuncs_C_A': [nn.L1Loss()],
    'lossNames_C_A': ['C_A_L1'],
    'lambdas_C_A': [100],
    'lossChannels_C_A': None,
    'lossFuncs_C_B': [nn.L1Loss()],
    'lossNames_C_B': ['C_B_L1'],
    'lambdas_C_B': [100],
    'lossChannels_C_B': None,
    'lr_g': 0.0002,
    'lr_d': 0.0001,
    'pretrained': None,
    'net_A': 'UNetSim3d',
    'net_B': 'UNetSim3d',    
    'channels_A': 1,
    'channels_B': 1,
    'step': 0,
    'num_iters': 100000,                  # change to num_steps
    'beta1': 0.5,
    'clip_max': 1,
    'imsize_A': [128,128,128],
    'super_resolution': 1,
    'adjust_lr_epoch': 20,
    'print_loss_freq_iter': 20,
    'save_cycleplot_freq_iter': 20,
    'save_val_freq_iter': 1000,
    'log_note': ' ',
    }

def train(**kwargs):
    kwargs = {**defaultkwargs, **kwargs}
    #opt = ParamOptions().parse()
    model = TrainModel(**kwargs)#(opt)

    model.init_model()
    dataset_size = len(model.train_dataset)
    #visualizer = Visualizer(opt)
    if model.pretrained != False:
        try:
            model.load_state(model.pretrained)
        except:
            model.load_state_old(model.pretrained)
        model.step = 0
        model.maxstep = 0
    for iteration in range(model.num_iters):
        i = iteration % model.num_samples
        model.normalize_lambdas()
        
        model.set_input(model.train_loader[i])
        
        if model.step < 150 and model.pretrained == False:
            model.optimizationE1(0.7)
        else:
            model.optimization()
            
        if model.step % 100 == 0:       ### HP optimizer
            model.save_lossdict()
            model.save_state()
            if model.step > 150:
                plotmetrics(f'{model.save_run}/losses.json')
                model.failtest(1.2)
        if model.step % model.print_loss_freq_iter == model.print_loss_freq_iter -1:
            losses = model.get_current_losses()
            model.print_current_losses(model.step,i,losses)
        if model.step % model.save_cycleplot_freq_iter == model.save_cycleplot_freq_iter -1:
            model.visual_iter(model.step,i)
        model.maxstep = max(model.maxstep,model.step)
        
        if model.step % model.save_val_freq_iter == model.save_val_freq_iter -1:
            with torch.no_grad():

                for k,test_data in enumerate(model.test_loader):
                    if k == 8:
                        break
                    model.val_input(test_data)
                    model.forward()
                    model.h5_val(model.step,name='test')
                    model.discr_val(model.step,name='test')
                    print(f'vset - {k}')
        model.step += 1
if __name__ == '__main__':
    train(num_samples=500)
