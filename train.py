import torch
from models.trainer import TrainModel
import os
import time
import json
from models.lossplot1 import plotmetrics

from torch import nn

defaultkwargs = {
    'run_path': F'{os.getcwd()}/results',           # path to save the results
    'run_name': time.strftime("%y_%m_%d-%H_%M"),    # folder name in run_path
    'step': 0,                                      # current training iteration - raise if you start with a pretrained network
    'num_iters': 100000,                            # total number of training iterations
    'num_samples': 10000,                           # number of randomly sampled image patches
    'batch_size': 4,                                
    'data_A': 'dataset/T700_GF_t05_pxs08.json',     # location of json file that specifies the datasets for domain A (fast)
    'data_B': 'dataset/T700_pco_pxs04.json',        # location of json file that specifies the datasets for domain B (slow)
    'lossFuncs_G_A': [nn.BCELoss()],                # list of loss functions for the generator loss A (result of discriminator B backpropagated to generator A(->B))
    'lossNames_G_A': ['G_A_BCE'],                   # list of unique names for the loss functions
    'lambdas_G_A': [10],                            # list of weights for the loss functions
    'lossChannels_G_A': None,                       # list of image channels that the loss functions are applied to
    'lossFuncs_G_B': [nn.BCELoss()],                # list of loss functions for the generator loss B (result of discriminator A backpropagated to generator B(->A))    
    'lossNames_G_B': ['G_B_BCE'],
    'lambdas_G_B': [10],
    'lossChannels_G_B': None,
    'lossFuncs_D_A': [nn.BCELoss()],                # list of loss functions for the discriminator loss A
    'lossNames_D_A': ['D_A_BCE'],
    'lambdas_D_A': [1],
    'lossChannels_D_A': None,
    'lossFuncs_D_B': [nn.BCELoss()],                # list of loss functions for the discriminator loss B
    'lossNames_D_B': ['D_B_BCE'],
    'lambdas_D_B': [1],
    'lossChannels_D_B': None,
    'lossFuncs_C_A': [nn.L1Loss()],                 # list of loss functions for the cycle consistency loss A
    'lossNames_C_A': ['C_A_L1'],
    'lambdas_C_A': [100],
    'lossChannels_C_A': None,
    'lossFuncs_C_B': [nn.L1Loss()],                 # list of loss functions for the cycle consistency loss B
    'lossNames_C_B': ['C_B_L1'],
    'lambdas_C_B': [100],
    'lossChannels_C_B': None,
    'lr_g': 0.0002,                                 # learning rate for generators
    'lr_d': 0.0001,                                 # learning rate for discriminators
    'pretrained': None,                             # path to pretrained networks - else None
    'HPoptimizer': True,                            # Do you want to use our dynamic hyperparameter updater? - It sucks but is better than manually choosing HPs :)
    'kickstart': True,                              # replaces generated images with input images during the first 150 steps. Prevents color inversions - only use if the intensities of both domains is similar
    'net_A': 'UNetSim3d',                           # generator network A->B - 'UNetSim3d', 'UNetSim3dSRx2', or 'UNetSim3dSRx4'
    'net_B': 'UNetSim3d',                           # generator network B->A - 'UNetSim3d', 'UNetSim3dSRd2', or 'UNetSim3dSRd4'
    'channels_A': 1,                                # number of (color-) channels in domain A
    'channels_B': 1,                                # number of (color-) channels in domain B
    'beta1': 0.5,
    'clip_max': 1,
    'imsize_A': [128,128,128],                      # training patch size in domain A
    'super_resolution': 1,                          # scaling factor from A to B - 1, 2, 4
    'adjust_lr_epoch': 20,
    'print_loss_freq_iter': 20,
    'save_cycleplot_freq_iter': 20,
    'save_val_freq_iter': 1000,
    'log_note': ' ',
    }



def train(**kwargs):
    kwargs = {**defaultkwargs, **kwargs}
    model = TrainModel(**kwargs)
    
    with open(model.save_log, 'w') as fp:
        json.dump(kwargs, fp, sort_keys=True, indent=4)

    model.init_model()
    dataset_size = len(model.train_dataset)
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
        
        if model.step < 150 and model.pretrained == False and model.kickstart == True:
            model.optimizationE1(0.7)
        else:
            model.optimization()
            
        if model.step % 100 == 0:       ### HP optimizer
            model.save_lossdict()
            model.save_state()
            if model.step > 150:
                plotmetrics(f'{model.save_run}/losses.json')
                if model.HPoptimizer == True:
                    model.failtest(1.)
        if model.step % model.print_loss_freq_iter == model.print_loss_freq_iter - 1:
            losses = model.get_current_losses()
            model.print_current_losses(model.step,i,losses)
        if model.step % model.save_cycleplot_freq_iter == model.save_cycleplot_freq_iter - 1:
            model.visual_iter(model.step,i)
        model.maxstep = max(model.maxstep, model.step)
        
        if model.step % model.save_val_freq_iter == model.save_val_freq_iter - 1:
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
    
## uncomment to train the 2x network for enhancement of the 1ms, 800nm dataset
    train(run_name = '2x_1ms',
        data_A = 'dataset/T700_GF_t05_pxs08.json',
        data_B = 'dataset/T700_GF_t05_pxs08.json',
        net_A = 'UNetSim3dSRx2',
        net_B = 'UNetSim3dSRd2',
        imsize_A = [128,128,128],
        super_resolution = 2)
    
## uncomment to train the 4x network for enhancement of the 3ms, 1600nm dataset
#     train(run_name = '4x_3ms',
#         data_A = 'dataset/T700_GF_t05_pxs08.json',
#         data_B = 'dataset/T700_GF_t05_pxs08.json',
#         net_A = 'UNetSim3dSRx4',
#         net_B = 'UNetSim3dSRx4',
#         imsize_A = [64,128,128],
#         super_resolution = 4)
    
## uncomment to train the 4x network for enhancement of the 1ms, 1600nm dataset
#     train(run_name = '4x_1ms',
#         data_A = 'dataset/T700_GF_t05_pxs08.json',
#         data_B = 'dataset/T700_GF_t05_pxs08.json',
#         net_A = 'UNetSim3dSRx4',
#         net_B = 'UNetSim3dSRx4',
#         imsize_A = [64,128,128],
#         super_resolution = 4)
    