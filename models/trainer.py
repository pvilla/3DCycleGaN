import itertools
import os
from torch import nn
from torch.nn.functional import interpolate
import torch
import numpy as np
import torchvision.transforms as transforms
from collections import OrderedDict
from abc import ABC
import models.networks as networks
from models.discriminator import NLayerDiscriminator3d
from models.initialization import init_weights
from models.data1channel import *
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.transform import resize
import h5py
import random
import json
import time
#matplotlib.use('TkAgg')


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def fixLossList(lossFuncs,lambdas=1,lossNames='loss',lossChannels=None):        #ensures that all loss-related lists are compatible
    l = len(lossFuncs)
    lambdas_f = []
    try:
        fl = float(lambdas)
        for i in range(l):
            lambdas_f.append(fl)
    except:
        for i in range(l):
            try:
                lambdas_f.append(float(lambdas[i]))
            except:
                lambdas_f.append(1.)
    lossNames_f = []
    if type(lossNames) ==str:
        for i in range(l):
            lossNames_f.append(f'{lossNames}_{i}')
    else:
        lnf = 'loss'
        for i in range(l):
            try:
                lossNames_f.append(str(lossNames[i]))
            except:
                lossNames_f.append(f'{lnf}_{i}')
    lossChannels_f = []
    try:
        lcf = int(lossChannels)
        for i in range(l):
            lossChannels_f.append([lcf])
    except:
        lcf = Ellipsis
        for i in range(l):
            try:
                lcfi = [int(lc) for lc in lossChannels[i]]
                lossChannels_f.append(lcfi)
            except:
                try:
                    lossChannels_f.append([int(lossChannels[i])])
                except:
                    lossChannels_f.append(lcf)
    return lambdas_f, lossNames_f, lossChannels_f


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=.1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice


class TrainModel(ABC):
    def __init__(self,**kwargs):
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self,key,value)
            
        
        self.fallbackcount = 0
        self.maxstep = 0
        
        self.lr_g_0 = self.lr_g
        self.lr_d_0 = self.lr_d
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device:{}'.format(self.device))
        self.model_init_type = 'normal'
        
        
        self.isTrain = True

        self.save_run = F"{self.run_path}/{self.run_name}"
        self.save_log = F"{self.save_run}/log.txt"
        self.save_stats = F"{self.save_run}/stats.txt"
        
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B']
        
        
        self.lambdas_G_A, self.lossNames_G_A, self.lossChannels_G_A= fixLossList(self.lossFuncs_G_A,lambdas=self.lambdas_G_A,lossNames=self.lossNames_G_A,lossChannels=self.lossChannels_G_A)
        self.lambdas_G_A_0 = np.copy(self.lambdas_G_A)

        self.lambdas_G_B, self.lossNames_G_B, self.lossChannels_G_B= fixLossList(self.lossFuncs_G_B,lambdas=self.lambdas_G_B,lossNames=self.lossNames_G_B,lossChannels=self.lossChannels_G_B)
        self.lambdas_G_B_0 = np.copy(self.lambdas_G_B)

        self.lambdas_D_A, self.lossNames_D_A, self.lossChannels_D_A= fixLossList(self.lossFuncs_D_A,lambdas=self.lambdas_D_A,lossNames=self.lossNames_D_A,lossChannels=self.lossChannels_D_A)
        self.lambdas_D_A_0 = np.copy(self.lambdas_D_A)

        self.lambdas_D_B, self.lossNames_D_B, self.lossChannels_D_B= fixLossList(self.lossFuncs_D_B,lambdas=self.lambdas_D_B,lossNames=self.lossNames_D_B,lossChannels=self.lossChannels_D_B)
        self.lambdas_D_B_0 = np.copy(self.lambdas_D_B)

        self.lambdas_C_A, self.lossNames_C_A, self.lossChannels_C_A= fixLossList(self.lossFuncs_C_A,lambdas=self.lambdas_C_A,lossNames=self.lossNames_C_A,lossChannels=self.lossChannels_C_A)
        self.lambdas_C_A_0 = np.copy(self.lambdas_C_A)

        self.lambdas_C_B, self.lossNames_C_B, self.lossChannels_C_B= fixLossList(self.lossFuncs_C_B,lambdas=self.lambdas_C_B,lossNames=self.lossNames_C_B,lossChannels=self.lossChannels_C_B)
        self.lambdas_C_B_0 = np.copy(self.lambdas_C_B)
        
        self.lrs={'lr_d':[],'lr_g':[]}
        self.losses={}
        self.lambdaLs={}
        
        if self.pretrained == None or self.pretrained == False:
            self.pretrained = False
            
        for l in [*self.lossNames_D_A,*self.lossNames_D_B,*self.lossNames_G_A,*self.lossNames_G_B,*self.lossNames_C_A,*self.lossNames_C_B]:
            self.losses[l]=[]
            self.lambdaLs[l]=[]
        print(self.losses)
        self.truthTables = {}
        self.ttNames = ['tt_D_A','tt_D_B']#,'tt_C_Bseg']
        for t in self.ttNames:
            self.truthTables[t] = {'TP': [], 'TN': [], 'FP': [], 'FN': []}
        
        self.imsize_B = self.imsize_A * self.super_resolution
        self.img_names = ['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B']
        print(f'{self.super_resolution}x super resolution')
    
    
    def lossList(self,lossFuncs,lambdas,lossChannels,lossNames,input,target):       # calculate loss with continous target (cycle losses)
        loss = 0
        for i in range(len(lossFuncs)):
            l = lossFuncs[i](input[:,lossChannels[i]],target[:,lossChannels[i]])
            loss += lambdas[i]*l
            self.losses[lossNames[i]].append(float(l))
            self.lambdaLs[lossNames[i]].append(float(lambdas[i]))
        return loss
    
    def lossListD(self,lossFuncs,lambdas,lossChannels,lossNames,t=None,f=None):     # calculate loss with binary target (GAN losses)
        loss = 0
        for i in range(len(lossFuncs)):
            l = 0
            if t is not None:
                target = self.logic_tensor(t, True)
                l += lossFuncs[i](t[:,lossChannels[i]],target[:,lossChannels[i]])
            if f is not None:
                target = self.logic_tensor(f, False)
                l += lossFuncs[i](f[:,lossChannels[i]],target[:,lossChannels[i]])
            loss += lambdas[i]*l
            self.losses[lossNames[i]].append(float(l))
            self.lambdaLs[lossNames[i]].append(float(lambdas[i]))
        return loss
    
    def lossf_G_A(self,input):
        target = self.logic_tensor(input, 1)
        return self.lossList(self.lossFuncs_G_A,self.lambdas_G_A,self.lossChannels_G_A,self.lossNames_G_A,input, target)
    
    def lossf_G_B(self,input):
        target = self.logic_tensor(input, 1)
        return self.lossList(self.lossFuncs_G_B,self.lambdas_G_B,self.lossChannels_G_B,self.lossNames_G_B,input, target)
    
    def lossf_D_A(self,t,f):
        return self.lossListD(self.lossFuncs_D_A,self.lambdas_D_A,self.lossChannels_D_A,self.lossNames_D_A,t,f)
    
    def lossf_D_B(self,t,f):
        return self.lossListD(self.lossFuncs_D_B,self.lambdas_D_B,self.lossChannels_D_B,self.lossNames_D_B,t,f)
    
    def lossf_C_A(self,input,target):
        return self.lossList(self.lossFuncs_C_A,self.lambdas_C_A,self.lossChannels_C_A,self.lossNames_C_A,input, target)
    
    def lossf_C_B(self,input,target):
        return self.lossList(self.lossFuncs_C_B,self.lambdas_C_B,self.lossChannels_C_B,self.lossNames_C_B,input, target)
    
    def tt_D(self,t,f,name):
        t = t > .5
        f = f > .5
        target = self.logic_tensor(t, True)
        self.truthTables[name]['TP'].append(int(torch.sum(t*target)))
        self.truthTables[name]['TN'].append(int(torch.sum(~f*target)))
        self.truthTables[name]['FP'].append(int(torch.sum(f*target)))
        self.truthTables[name]['FN'].append(int(torch.sum(~t*target)))

    # def criterionCycle_A(self,input,target):
    #     loss = 0
    #     for i in range(len(self.lambda_A)):
    #         loss += self.lambda_A[i]*self.criterionCycleList_A[i](input[:,i:i+1,...],target[:,i:i+1,...])
    #     return loss
    
    # def criterionCycle_B(self,input,target):
    #     loss = 0
    #     for i in range(len(self.lambda_B)):
    #         l = self.lambda_B[i]*self.criterionCycleList_B[i](input[:,i:i+1,...],target[:,i:i+1,...])
    #         loss += l
    #         #print(i,l)
    #     return loss
    
    # def append_current_losses(self):
    #     errors_list = OrderedDict()
    #     for name in self.loss_names:
    #         if isinstance(name, str):
    #             errors_list[name] = float(
    #                 getattr(self, 'loss_' + name))
    #     return errors_list
        

    def get_current_losses(self):
        errors_list = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_list[name] = float(
                    getattr(self, 'loss_' + name))
        return errors_list

    def print_current_losses(self, epoch, iters, losses):
        message = 'Epoch [{}/{}], Step [{}/{}]'.format(epoch+1, self.num_epochs, iters+1, self.stepN_epoch)
        for name, loss in losses.items():
            message += ', {:s}: {:.3f}'.format(name, loss)
        print(message)
        with open(self.save_log, "a") as f:
            print(message,file=f)
        
    def save_lossdict(self,name='losses'):
        ldict = {'losses': self.losses,
                 'lambdas': self.lambdaLs,
                 'lrs': self.lrs,
                 'tts': self.truthTables,
                 'misc': self.log_note}
        
        with open(f'{self.save_run}/{name}.json', 'w') as fp:
            json.dump(ldict, fp, sort_keys=True, indent=4, skipkeys=True)

    def adjust_learning_rate(self, epoch, optimizer, initial_lr):
        """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
        lr = initial_lr * (0.1 ** (epoch // self.adjust_lr_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def adjust_learning_rate_D(self, epoch, optimizer, initial_lr):
        """Sets the learning rate to the initial LR decayed by 10 every X epochs"""
        standard_lr = initial_lr * (0.1 ** (epoch // self.adjust_lr_epoch))
        if epoch == 0:
            lr = standard_lr
        else:
            lr = current_lr = optimizer.param_groups[-1]['lr']
            loss_D = torch.mean(torch.tensor([self.loss_D_A,self.loss_D_B]))
            print('ćurrent_lr',current_lr)
            if epoch % self.adjust_lr_epoch == self.adjust_lr_epoch -1:
                lr = 0.1 * lr
            if loss_D < 0.45:
                lr = np.sqrt(0.1) * lr
            elif loss_D > 0.8:
                lr = np.sqrt(10) * lr
            
            print('standard_lr',standard_lr)
            print('current_lr',current_lr)
            print('new_lr',lr)
            
            
        for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    def update_learning_rate(self,epoch):
        self.adjust_learning_rate(epoch, self.optimizer_G,self.lr_g)
        #self.adjust_learning_rate_D(epoch, self.optimizer_D,self.lr_d)
        self.adjust_learning_rate(epoch, self.optimizer_D, self.lr_d)
        

            
    def load_data(self):
        print('start loading data....')
            
        self.train_dataset = Dataset3d(dfile_A = self.data_A, dfile_B = self.data_B, settype = 'train', dsize_A = self.imsize_A, dsize_B = self.imsize_B, crop = 'random', dlen = self.num_samples)
        train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=4,pin_memory=True)

        ################# increase num workers

        test_dataset = Dataset3d(dfile_A = self.data_A, dfile_B = self.data_B, settype = 'validate', dsize_A = self.imsize_A, dsize_B = self.imsize_B, crop = 'random', dlen = self.num_samples)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)
        print('finish loading data')
        return train_loader, test_loader


    def get_model_A(self, num_out=1, num_in=1):
        model = getattr(networks,self.net_A)(num_out = num_out, num_in = num_in)#S(pretrained=self.pretrained, num_out=num_out)
        model.eval()
        # print('1')
        if self.pretrained is not True:
            init_weights(model, self.model_init_type, init_gain=0.02)
        return model
    
    def get_model_B(self, num_out=1, num_in=1):
        model = getattr(networks,self.net_B)(num_out = num_out, num_in = num_in)#(pretrained=self.pretrained, num_out=num_out)
        model.eval()
        # print('1')
        if self.pretrained is not True:
            init_weights(model, self.model_init_type, init_gain=0.02)
        return model

    def get_NLdnet(self, num_input=1):
        dnet = NLayerDiscriminator3d(num_input,n_layers=3)
        dnet.eval()
        init_weights(dnet, self.model_init_type, init_gain=0.02)
        return dnet

    def create_dir_if_not_exist(self, path):
        if os.path.exists(path):
            # decision = input('This folder already exists. Continue training will overwrite the data. Proceed(y/n)?')
            # if decision != 'y':
            #     exit()
            # else:
                print(f'Warning! Overwriting folder: {self.run_name}  in 10s.')
                time.sleep(10)
                
        if not os.path.exists(path):
            os.makedirs(path)

    def init_model(self):

        self.netG_A = nn.DataParallel(self.get_model_A(num_out=self.channels_B,num_in=self.channels_A)).to(self.device)
        self.netG_B = nn.DataParallel(self.get_model_B(num_out=self.channels_A,num_in=self.channels_B)).to(self.device)
        self.netD_A = nn.DataParallel(self.get_NLdnet(self.channels_B)).to(self.device)
        self.netD_B = nn.DataParallel(self.get_NLdnet(self.channels_A)).to(self.device)
            
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=self.lr_g, betas=(self.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=self.lr_d, betas=(self.beta1, 0.999))
        self.train_loader, self.test_loader = self.load_data()
        self.stepN_epoch = len(self.train_loader)
        self.create_dir_if_not_exist(self.save_run)

    def optimize_parameters(self):
        self.init_model()

    def set_input(self,input):
            self.images, self.attenuation = [input[i].to(self.device,dtype=torch.float) for i in range(2)]
            self.r_index = torch.randperm(self.images.shape[0])
            self.real_A = self.images.to(self.device)
            self.real_B = self.attenuation[self.r_index][:,:]#.unsqueeze(1)


    def val_input(self,input):
            self.images, self.attenuation = [input[i].to(self.device,dtype=torch.float) for i in range(2)]
            self.r_index = torch.arange(self.images.shape[0])
            self.real_A = self.images.to(self.device)
            self.real_B = self.attenuation[self.r_index][:,:]#.unsqueeze(1)   



    def GANLoss(self, pred, is_real):
        target = self.logic_tensor(pred, is_real)
        loss = self.criterionBSE(pred, target)
        return loss

    def FRCLoss(self, img1, img2):      #not implimented at the moment
        if len(img1.shape) > 3:
            img1,img2 = img1.squeeze(1),img2.squeeze(1)
        nz,nx,ny= [torch.tensor(i, device=self.device) for i in img1.shape]
        rnyquist = nx//2
        x = torch.cat((torch.arange(0, nx / 2), torch.arange(-nx / 2, 0))).to(self.device)
        y = x
        X, Y = torch.meshgrid(x, y)
        map = X ** 2 + Y ** 2
        index = torch.round(torch.sqrt(map.float()))
        r = torch.arange(0, rnyquist + 1).to(self.device)
        F1 = torch.rfft(img1, 2, onesided=False).permute(1, 2, 0, 3)
        F2 = torch.rfft(img2, 2, onesided=False).permute(1, 2, 0, 3)
        C_r,C1,C2,C_i = [torch.empty(rnyquist + 1, self.batch_size).to(self.device) for i in range(4)]
        for ii in r:
            auxF1 = F1[torch.where(index == ii)]
            auxF2 = F2[torch.where(index == ii)]
            C_r[ii] = torch.sum(auxF1[:, :, 0] * auxF2[:, :, 0] + auxF1[:, :, 1] * auxF2[:, :, 1], axis=0)
            C_i[ii] = torch.sum(auxF1[:, :, 1] * auxF2[:, :, 0] - auxF1[:, :, 0] * auxF2[:, :, 1], axis=0)
            C1[ii] = torch.sum(auxF1[:, :, 0] ** 2 + auxF1[:, :, 1] ** 2, axis=0)
            C2[ii] = torch.sum(auxF2[:, :, 0] ** 2 + auxF2[:, :, 1] ** 2, axis=0)

        FRC = torch.sqrt(C_r ** 2 + C_i ** 2) / torch.sqrt(C1 * C2)
        FRCm = 1 - torch.where(FRC != FRC, torch.tensor(1.0, device=self.device), FRC)
        My_FRCloss = torch.mean((FRCm) ** 2)
        return My_FRCloss

    def plot_cycle_cyclegan(self, img_idx,save_name, layer=0, test=False, plot_phase=True):
        """set layer to 1 and plot_phase to False to plot imag channel"""
        img_list = ['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B']
        if plot_phase is True:
            img_list[1], img_list[3], img_list[5] = [img_list[i] + '_ph' for i in [1, 3, 5]]
        
        fig, axs = plt.subplots(2, int(len(img_list) / 2), figsize=(20, 20), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace=0.0001, wspace=0.0001)
        
        vmin = -3
        vmax = 3
        axs = axs.ravel()
        ignore = [1, 3, 5]  # list of indices to be ignored
        l = [idx for idx in range(len(img_list)) if idx not in ignore]
        for i in l:
            if test == True:
                im = axs[i].imshow(getattr(self,img_list[i])[img_idx, 0, 15, :, :].cpu(),vmin=vmin,vmax=vmax)
            else:
                im = axs[i].imshow(getattr(self,img_list[i])[img_idx, 0, 15, :, :].detach().cpu(),vmin=vmin,vmax=vmax)
            axs[i].axis("off")
            axs[i].set_title(img_list[i], fontsize=36)
        for i in ignore:
            if test == True:
                im = axs[i].imshow(getattr(self,img_list[i])[img_idx, layer, 15, :, :].cpu(),vmin=vmin,vmax=vmax)
            else:
                im = axs[i].imshow(getattr(self,img_list[i])[img_idx, layer, 15, :, :].detach().cpu(),vmin=vmin,vmax=vmax)
            axs[i].axis("off")
            axs[i].set_title(img_list[i], fontsize=36)
        if save_name != 0:
            if test is True:
                save_path = F"{self.save_run}/test"
            else:
                save_path = F"{self.save_run}/train"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(save_path + F"/{save_name}.png")
            plt.cla()
            plt.close()

    def logic_tensor(self, pred, is_real):
        if is_real:
            target = torch.tensor(1.0)
        else:
            target = torch.tensor(0.0)
        return target.expand_as(pred).to(self.device)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def backward_D(self, netD, real, fake, rec=None):
        
        pred_real = netD(real)
        loss_D_real = self.GANLoss(pred_real, True)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.GANLoss(pred_fake, False)
        
        ######NEW
        if rec is not None:
            
            pred_rec = netD(rec.detach())
            loss_D_rec = self.GANLoss(pred_rec, False)        
            loss_D = (loss_D_real + loss_D_fake + loss_D_rec) * 0.3333
        ######NEW
        else:
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D
    
    def backward_D_A(self, real, fake):
        pred_real = self.netD_A(real)
        pred_fake = self.netD_A(fake.detach())
        #print(pred_real)
        loss = self.lossf_D_A(pred_real,pred_fake)
        self.tt_D(pred_real, pred_fake, 'tt_D_A')
        loss.backward()
        return loss
    
    def backward_D_B(self, real, fake):
        pred_real = self.netD_B(real)
        pred_fake = self.netD_B(fake.detach())
        #print(pred_real)
        loss = self.lossf_D_B(pred_real,pred_fake)
        self.tt_D(pred_real, pred_fake, 'tt_D_B')
        loss.backward()
        return loss
    
    def backward_G(self):
        self.loss_G_A = self.lossf_G_A(self.netD_A(self.fake_B))
        self.loss_cycle_B = self.lossf_C_B(self.rec_B, self.real_B)
        self.loss_G_B = self.lossf_G_B(self.netD_B(self.fake_A))
        self.loss_cycle_A = self.lossf_C_A(self.rec_A, self.real_A)
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B
        self.loss_G.backward()
        if self.clip_max != 0:
            nn.utils.clip_grad_norm_(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), self.clip_max)
    

    def forward(self):        
        #self.fake_B = self.netG_A(self.real_A)
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)
        
        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

    def forwardE1(self,frp = 0.2,seg_threshold=0,seg_pm=-1): # randomly exchange fakes with interpolated reals - Suppresses color flips
        #self.fake_B = self.netG_A(self.real_A)
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)
        
        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)
        
        rn = np.random.random()
        if rn < frp:
            fakefake_B = interpolate(self.real_A,self.fake_B.size()[-3:])
            self.rec_A = self.netG_B(fakefake_B)
            
            fakefake_A = interpolate(self.real_B,self.fake_A.size()[-3:])
            self.rec_B = self.netG_A(fakefake_A)
                
    def binarizeChannel(self, tensor, channel0 = 1, channel1 = -1, threshold = 0):
        tensor[:,channel0:channel1,...] = tensor[:,channel0:channel1,...] > threshold
        return tensor                

    def optimization(self):
        # with torch.cuda.amp.autocast():
        self.forward()
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.loss_D_A = self.backward_D_A(self.real_B, self.fake_B)
        self.loss_D_B = self.backward_D_B(self.real_A, self.fake_A)
        self.optimizer_D.step()
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.lrs['lr_g'].append(self.lr_g)
        self.lrs['lr_d'].append(self.lr_d)
        
        
    def optimizationE1(self,frp = 0.2,seg_threshold=0,seg_pm=-1):
        # with torch.cuda.amp.autocast():
        self.forwardE1(frp = frp,seg_threshold=seg_threshold,seg_pm=seg_pm)
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.loss_D_A = self.backward_D_A(self.real_B, self.fake_B)
        self.loss_D_B = self.backward_D_B(self.real_A, self.fake_A)
        self.optimizer_D.step()
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.lrs['lr_g'].append(self.lr_g)
        self.lrs['lr_d'].append(self.lr_d)

    def write_to_stat(self, epoch,iter):
        with open(self.save_stats, "a+") as f:
            f.write('\n -------------------------------------------------------\nEpoch [{}/{}], Step [{}/{}]\n'.format(
                epoch + 1, self.num_epochs, iter + 1, self.stepN_epoch))
            for i in range(len(self.img_names)):
                self.print_numpy_to_log(getattr(self,self.img_names[i]).detach().cpu().numpy(), f, self.img_names[i])

    def save_net(self, name,epoch,net,optimizer,loss):
        model_save_name = F'{self.run_name}_{name}_{epoch}ep.pt'
        path = F"{self.save_run}/save"
        if not os.path.exists(path):
            os.makedirs(path)
        print('saving trained model {}'.format(model_save_name))
        torch.save({
          'epoch': epoch,
          'model_state_dict': net.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'loss': loss}, path+F'/{model_save_name}')
    


    def print_numpy_to_log(self, x, f, note):
        x = x.astype(np.float64)
        x = x.flatten()
        # print('%s:  mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (note,np.mean(x), np.min(x),np.max(x),np.median(x), np.std(x)),file=f)

    def visual_iter(self,epoch,iter):
        self.write_to_stat(epoch, iter)
        save_name = f'cycle_plot_step{self.step}'
        self.plot_cycle_cyclegan(0,save_name,0,False,plot_phase=False)
        self.plot_cycle_h5(save_name)
            
    def plot_cycle_h5(self,name='train'):
        path = F"{self.save_run}/eval"
        if not os.path.exists(path):
            os.makedirs(path)
        save_name = '{}/train/{}.h5'.format(self.save_run,name)
        #print(self.real_A.size(),self.fake_B.size(),self.rec_A.size())
        real_A = self.real_A
        fake_A = self.fake_A
        rec_A = self.rec_A
        real_B = self.real_B
        fake_B = self.fake_B
        rec_B = self.rec_B
        if self.super_resolution:
            real_A = interpolate(real_A, scale_factor=self.super_resolution, mode='nearest')
            fake_A = interpolate(fake_A, scale_factor=self.super_resolution, mode='nearest')
            rec_A = interpolate(rec_A, scale_factor=self.super_resolution, mode='nearest')
        ABA = torch.cat((real_A,fake_B,rec_A),dim=1)
        ABA = ABA.detach().cpu().numpy()
        BAB = torch.cat((real_B,fake_A,rec_B),dim=1)
        BAB = BAB.detach().cpu().numpy()
        with h5py.File(save_name,'a') as f:
            if 'ABA' in f:
                f['ABA'].resize((ABA.shape[0]+f['ABA'].shape[0]),axis=0)
                f['ABA'][-ABA.shape[0]:] = ABA
            else:
                mshape = (None,*ABA.shape[1:])
                f.create_dataset('ABA', shape=ABA.shape,dtype = 'float32',data = ABA, maxshape = mshape)
                
            if 'BAB' in f:
                f['BAB'].resize((BAB.shape[0]+f['BAB'].shape[0]),axis=0)
                f['BAB'][-BAB.shape[0]:] = BAB
            else:
                mshape = (None,*BAB.shape[1:])
                f.create_dataset('BAB', shape=BAB.shape,dtype = 'float32',data = BAB, maxshape = mshape)

    def visual_val(self,epoch,idx):
        save_name = '{:03d}epoch_{:02d}'.format(epoch + 1, idx+1)
        
        self.plot_cycle_cyclegan(0,save_name,0,True)
            
    def h5_val(self,epoch,name='val'):
        path = F"{self.save_run}/eval"
        if not os.path.exists(path):
            os.makedirs(path)
        save_name = '{}/eval/{}_{:03d}epoch.h5'.format(self.save_run,name,epoch)
        # print(self.real_A.size(),self.fake_B.size(),self.rec_A.size())
        real_A = self.real_A
        fake_A = self.fake_A
        rec_A = self.rec_A
        real_B = self.real_B
        fake_B = self.fake_B
        rec_B = self.rec_B
        if self.super_resolution:
            real_A = interpolate(real_A, scale_factor=self.super_resolution, mode='nearest')
            fake_A = interpolate(fake_A, scale_factor=self.super_resolution, mode='nearest')
            rec_A = interpolate(rec_A, scale_factor=self.super_resolution, mode='nearest')
        ABA = torch.cat((real_A,fake_B,rec_A),dim=1)
        ABA = ABA.detach().cpu().numpy()
        BAB = torch.cat((real_B,fake_A,rec_B),dim=1)
        BAB = BAB.detach().cpu().numpy()
        with h5py.File(save_name,'a') as f:
            if 'ABA' in f:
                f['ABA'].resize((ABA.shape[0]+f['ABA'].shape[0]),axis=0)
                f['ABA'][-ABA.shape[0]:] = ABA
            else:
                mshape = (None,*ABA.shape[1:])
                f.create_dataset('ABA', shape=ABA.shape,dtype = 'float32',data = ABA, maxshape = mshape)
                
            if 'BAB' in f:
                f['BAB'].resize((BAB.shape[0]+f['BAB'].shape[0]),axis=0)
                f['BAB'][-BAB.shape[0]:] = BAB
            else:
                mshape = (None,*BAB.shape[1:])
                f.create_dataset('BAB', shape=BAB.shape,dtype = 'float32',data = BAB, maxshape = mshape)
                
    def h5_fake_B(self,epoch,name='fake_B'):
        path = F"{self.save_run}/eval"
        if not os.path.exists(path):
            os.makedirs(path)
        save_name = '{}/eval/{}_{:03d}epoch.h5'.format(self.save_run,name,epoch)
        # print(self.fake_B.size())
        fake_B = self.fake_B.cpu()
        with h5py.File(save_name,'a') as f:
            if 'fake_B' in f:
                f['fake_B'].resize((fake_B.shape[0]+f['fake_B'].shape[0]),axis=0)
                f['fake_B'][-fake_B.shape[0]:] = fake_B
            else:
                mshape = (None,*fake_B.shape[1:])
                f.create_dataset('fake_B', shape=fake_B.shape,dtype = 'float32',data = fake_B, maxshape = mshape)
                

    def discr_val(self,epoch,name='val'):
        path = F"{self.save_run}/eval"
        if not os.path.exists(path):
            os.makedirs(path)
        save_A_B = '{}/eval/{}_DAB_{:03d}epoch.npy'.format(self.save_run,name,epoch)
        save_B_A = '{}/eval/{}_DBA_{:03d}epoch.npy'.format(self.save_run,name,epoch)
        
        D_A_real_B = self.netD_A(self.real_B).detach().cpu().numpy()
        D_A_real = [np.sum(np.round(D_A_real_B)),D_A_real_B.size]
        D_A_fake_B = self.netD_A(self.fake_B).detach().cpu().numpy()
        D_A_fake = [np.sum(np.round(D_A_fake_B)),D_A_fake_B.size]
        D_A_rec_B = self.netD_A(self.rec_B).detach().cpu().numpy()
        D_A_rec= [np.sum(np.round(D_A_rec_B)),D_A_rec_B.size]
        D_A_B = np.stack([D_A_real,D_A_fake,D_A_rec])
        
        D_B_real_A = self.netD_B(self.real_A).detach().cpu().numpy()
        D_B_real = [np.sum(np.round(D_B_real_A)),D_B_real_A.size]
        D_B_fake_A = self.netD_B(self.fake_A).detach().cpu().numpy()
        D_B_fake = [np.sum(np.round(D_B_fake_A)),D_B_fake_A.size]
        D_B_rec_A = self.netD_B(self.rec_A).detach().cpu().numpy()
        D_B_rec= [np.sum(np.round(D_B_rec_A)),D_B_rec_A.size]
        D_B_A = np.stack([D_B_real,D_B_fake,D_B_rec])
        
        try:
            with open(save_A_B,'rb') as f_A_B, open(save_B_A,'rb') as f_B_A:
                A_B = np.load(f_A_B)
                A_B = np.concatenate([A_B,np.expand_dims(D_A_B,0)],0)
                B_A = np.load(f_B_A)
                B_A = np.concatenate([B_A,np.expand_dims(D_B_A,0)],0)
        except:
            A_B = np.expand_dims(D_A_B,0)
            B_A = np.expand_dims(D_B_A,0)
        # print(D_A_B,D_B_A)
        with open(save_A_B,'wb') as f_A_B, open(save_B_A,'wb') as f_B_A:
            np.save(f_A_B,A_B)
            np.save(f_B_A,B_A)
            
    def save_models(self, epoch):
        self.save_net('netG_A', epoch + 1, self.netG_A, self.optimizer_G, self.loss_G_A)
        self.save_net('netG_B', epoch + 1, self.netG_B, self.optimizer_G, self.loss_G_B)
        self.save_net('netD_A', epoch + 1, self.netD_A, self.optimizer_D, self.loss_D_A)
        self.save_net('netD_B', epoch + 1, self.netD_B, self.optimizer_D, self.loss_D_B)

    def load_trained_model(self, load_path, load_epoch):
        net_name = F'{load_path.split("/")[-1]}_netG_A_{load_epoch}ep.pt'
        path = F"{load_path}/save/{net_name}"
        # optimizer = torch.optim.Adam(net.parameters())
        checkpoint = torch.load(path)
        self.netG_A.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        self.netG_A.eval()
        
    def load_netG_A(self, path):
        checkpoint = torch.load(path)
        self.netG_A.load_state_dict(checkpoint['model_state_dict'])
        self.netG_A.eval()
        
    def load_pretrained_models(self,load_path,load_epoch):
        netD_A_name = F'{load_path.split("/")[-1]}_netD_A_{load_epoch}ep.pt'
        print(F'{load_path.split("/")[-1]}_netD_A_{load_epoch}ep.pt')
        netD_B_name = F'{load_path.split("/")[-1]}_netD_B_{load_epoch}ep.pt'
        netG_A_name = F'{load_path.split("/")[-1]}_netG_A_{load_epoch}ep.pt'
        netG_B_name = F'{load_path.split("/")[-1]}_netG_B_{load_epoch}ep.pt'
        checkpointD_A = torch.load(netD_A_name)
        checkpointD_B = torch.load(netD_B_name)
        checkpointG_A = torch.load(netG_A_name)
        checkpointG_B = torch.load(netG_B_name)
        self.netD_A.load_state_dict(checkpointD_A['model_state_dict'])
        self.netD_B.load_state_dict(checkpointD_B['model_state_dict'])
        self.netG_A.load_state_dict(checkpointG_A['model_state_dict'])
        self.netG_B.load_state_dict(checkpointG_B['model_state_dict'])
        
##### HP trainer
    
    def fliptest(self):         #compares fakes to reals and flats. if a flaw is detected a True value is returned.
        real_A = self.real_A[:, 0, ...].detach().cpu().numpy()
        real_A = np.moveaxis(real_A,0,-1)
        real_A = resize(real_A,(64,64,64))
        fake_B = self.fake_B[:, 0, ...].detach().cpu().numpy()
        fake_B = np.moveaxis(fake_B,0,-1)
        fake_B = resize(fake_B,(64,64,64))
        mean_fake_B = np.ones_like(fake_B)*np.mean(fake_B)
        
        real_B = self.real_B[:, 0, ...].detach().cpu().numpy()
        real_B = np.moveaxis(real_B,0,-1)
        real_B = resize(real_B,(64,64,64))
        fake_A = self.fake_A[:, 0, ...].detach().cpu().numpy()
        fake_A = np.moveaxis(fake_A,0,-1)
        fake_A = resize(fake_A,(64,64,64))
        mean_fake_A = np.ones_like(fake_A)*np.mean(fake_A)
        
        loss_rAfB = mse(real_A,fake_B)
        loss_nrAfB = mse(-1*real_A,fake_B)
        loss_mBfB = mse(mean_fake_B,fake_B)
        
        loss_rBfA = mse(real_B,fake_A)
        loss_nrBfA = mse(-1*real_B,fake_A)
        loss_mAfA = mse(mean_fake_A,fake_A)
        
        print('loss_rAfB',loss_rAfB,'  loss_nrAfB',loss_nrAfB,'  loss_mBfB',loss_mBfB)
        print('loss_rBfA',loss_rBfA,'  loss_nrBfA',loss_nrBfA,'  loss_mAfA',loss_mAfA)
        
        if loss_rAfB > loss_nrAfB and loss_rBfA > loss_nrBfA:
            print('flip detected!')
            return 20
        elif loss_rAfB > loss_nrAfB or loss_rBfA > loss_nrBfA:
            print('partial flip detected!')
            return .5
        elif loss_rAfB > loss_mBfB or loss_rBfA > loss_mAfA:
            print('partial flat detected! continue training!')
            return 0
        else:
            print('no faults detected! continue training!')
            return 0

    def xLosstest(self):        #check for exploding losses
        N = 20
        xl = 0
        for key in self.losses.keys():
            d = self.losses[key][:]
            rmean = np.convolve(d, np.ones(N)/N, mode='valid')
            rmean200 = np.convolve(d[-200:], np.ones(N)/N, mode='valid')
            min_rmean = np.min(rmean)
            max_rmean200 = np.max(rmean200)
            mean0 = np.mean(d[-200:-100])
            mean1 = np.mean(d[-100:])
            if (4*mean0) < max_rmean200:
                xp = (max_rmean200/(3*mean0))-1
                xl += xp
                print(f'explosion detected in {key} - {xp}')
            if mean1 > mean0:
                xp = (mean1/mean0)-1
                xl += xp
                print(f'increase detected in {key} - {xp}')
            if mean1 > 20:
                xl += 20
                print(f'fail detected in {key} - 20')
        print(f'xloss = {xl}')
        return xl

    def cLosstest(self):        #check for drifting cycle losses
        N = 20
        xl = 0
        for key in [*self.lossNames_C_A,*self.lossNames_C_B]:
            d = self.losses[key][:]
            rmean = np.convolve(d, np.ones(N)/N, mode='valid')
            min_rmean = np.min(rmean)
            mean1 = np.mean(d[-100:])
            if mean1 > (min_rmean+.1)*1.4:
                xl = 20
                print(f'cycle limit exceeded in {key}')
        return xl
    
    def fix_lrs(self,skewfac = 0.5,dfac=1e-4,gfac=1e-3):
        print(f'old lr_d = {self.lr_d}, old lr_g = {self.lr_g}')
        self.lr_d = 0.5 * (self.lr_d + dfac * (1-skewfac)**2 * 0.99998**self.step)
        self.lr_g = 0.5 * (self.lr_g + gfac * (skewfac)**2 * 0.99998**self.step)
        print(f'new lr_d = {self.lr_d}, new lr_g = {self.lr_g}')
        
    
    def adjust_lambda(self, lambda_in, sign = 1,omin = .1, omax = 100, factor = 0.3):
        lambda_out = lambda_in + .1      # add 1 to avoid zero lambdas
        if sign < 0:
            lambda_out *= 1 - factor
        else:
            lambda_out *= 1 + factor
        return max(min(lambda_out,omax),omin)
    
    def rand_lambdas(self):
        #cycle A
        for i in range(len(self.lambdas_C_A)):
            self.lambdas_C_A[i] = self.adjust_lambda(self.lambdas_C_A[i], factor = np.random.rand()-.5)
            
        #cycle B
        for i in range(len(self.lambdas_C_B)):
            self.lambdas_C_B[i] = self.adjust_lambda(self.lambdas_C_B[i], factor = np.random.rand()-.5)
           
################## Add multi loss support for gan loss!!!!!!!!!!!!!!!!
        #gen A
        for i in range(len(self.lambdas_G_A)):
            self.lambdas_G_A[i] = self.adjust_lambda(self.lambdas_G_A[i], factor = np.random.rand()-.5)
            
        #gen B
        for i in range(len(self.lambdas_G_B)):
            self.lambdas_G_B[i] = self.adjust_lambda(self.lambdas_G_B[i], factor = np.random.rand()-.5)
            
        #dis A
        for i in range(len(self.lambdas_D_A)):
            self.lambdas_D_A[i] = self.adjust_lambda(self.lambdas_D_A[i], factor = np.random.rand()-.5)
            
        #dis B
        for i in range(len(self.lambdas_D_B)):
            self.lambdas_D_B[i] = self.adjust_lambda(self.lambdas_D_B[i], factor = np.random.rand()-.5)
            
        self.normalize_lambdas(200)
                
    def fix_lambdas(self):
        ctarget = self.cycle_target([*self.lossNames_C_A,*self.lossNames_C_B])
        #cycle A
        for i in range(len(self.lambdas_C_A)):
            grad = self.loss_grad(self.lossNames_C_A[i])
            sign = grad - ctarget
            print(f'{self.lossNames_C_A[i]} - sign {sign}')
            self.lambdas_C_A[i] = self.adjust_lambda(self.lambdas_C_A[i], sign = sign)
            
        #cycle B
        for i in range(len(self.lambdas_C_B)):
            grad = self.loss_grad(self.lossNames_C_B[i])
            sign = grad - ctarget
            print(f'{self.lossNames_C_B[i]} - sign {sign}')
            self.lambdas_C_B[i] = self.adjust_lambda(self.lambdas_C_B[i], sign = sign)
           
################## Add multi loss support for gan loss!!!!!!!!!!!!!!!!
        #gen A
        for i in range(len(self.lambdas_G_A)):
            ttmean = self.tt_mean('tt_D_A','FP')
            self.lambdas_G_A[i] = self.adjust_lambda(self.lambdas_G_A[i], sign = 0.4 - ttmean)
            
        #gen B
        for i in range(len(self.lambdas_G_B)):
            ttmean = self.tt_mean('tt_D_B','FP')
            self.lambdas_G_B[i] = self.adjust_lambda(self.lambdas_G_B[i], sign = 0.4 - ttmean)
            
        #dis A
        for i in range(len(self.lambdas_D_A)):
            ttmean = self.tt_mean('tt_D_A')
            self.lambdas_D_A[i] = self.adjust_lambda(self.lambdas_D_A[i], sign = 0.6 - ttmean)
            
        #dis B
        for i in range(len(self.lambdas_D_B)):
            ttmean = self.tt_mean('tt_D_B')
            self.lambdas_D_B[i] = self.adjust_lambda(self.lambdas_D_B[i], sign = 0.6 - ttmean)
            
        self.normalize_lambdas(200)
        
    def ll_0(self):
        self.lambdas_C_A = np.copy(self.lambdas_C_A_0)
        self.lambdas_C_B = np.copy(self.lambdas_C_B_0)
        self.lambdas_G_A = np.copy(self.lambdas_G_A_0)
        self.lambdas_G_B = np.copy(self.lambdas_G_B_0)
        self.lambdas_D_A = np.copy(self.lambdas_D_A_0)
        self.lambdas_D_B = np.copy(self.lambdas_D_B_0)
        
        
        self.normalize_lambdas(200)
        
        self.lr_d = self.lr_d_0
        self.lr_g = self.lr_g_0
        
    def normalize_lambdas(self,threshold = 200):
        llist = ['lambdas_C_A','lambdas_C_B','lambdas_G_A','lambdas_G_B','lambdas_D_A','lambdas_D_B']
        total = 0
        for ln in llist:
            lams = getattr(self,ln)
            for lam in lams:
                total += lam
                
        nlstr = ''
                
        for ln in llist:
            lams = getattr(self,ln)
            for i in range(len(lams)):
                lams[i] = threshold * lams[i] / total
            nlstr = nlstr + f', {ln}: {lams}'
            setattr(self,ln,lams)
        print(f'lambdas updated{nlstr}')
        
            

    def tt_mean(self,tt_name = 'tt_D_A',column = None, past = 100):     # column is the type (TP, TN, FP, FN) else TP+TN
        tt = self.truthTables[tt_name]
        total = 0
        for key in tt.keys():
            total += tt[key][0]
        if column in tt.keys():
            tpast = 2*np.mean(tt[column][-past:])
        else:
            tpast = np.mean(tt['TP'][-past:]) + np.mean(tt['TN'][-past:])
        print(column,tpast/total)
        return (tpast/total)
        
    
    def loss_grad(self,loss_name):
        d = self.losses[loss_name][-200:]
        x = np.arange(len(d))
        grad,b = np.polyfit(x,d,1)
        return grad

    def cycle_target(self,loss_names):
        grads = []
        for key in loss_names:
            grads.append(self.loss_grad(key))
        grads = np.array(grads)
        grads = np.abs(grads)
        return -np.mean(grads)
    
    def save_state(self):
        step = np.round(self.step,-2)
        model_save_name = F'{self.run_name}_{step:06}.pt'
        path = F"{self.save_run}/save"
        if not os.path.exists(path):
            os.makedirs(path)
        fullpath = F'{path}/{model_save_name}'
        print('saving trained model {}'.format(model_save_name))
        
        nets = {'netG_A':self.netG_A.state_dict(),
                'netG_B':self.netG_B.state_dict(),
                'netD_A':self.netD_A.state_dict(),
                'netD_B':self.netD_B.state_dict()}
        
        optimizers = {'optimizer_G':self.optimizer_G.state_dict(),
                      'optimizer_D':self.optimizer_D.state_dict()}
        
        losses = {'loss_G_A':self.loss_G_A,
                  'loss_G_B':self.loss_G_B,
                  'loss_D_A':self.loss_D_A,
                  'loss_D_B':self.loss_D_B}
        
        torch.save({
          'step': self.step,
          'model_state_dict': nets,
          'optimizer_state_dict': optimizers,
          'loss': losses}, fullpath)

        
    
    def load_state(self, fname, step = None):
        checkpoint = torch.load(fname)
        if step == None:
            self.step = checkpoint['step']
        else:
            step = step
        models = checkpoint['model_state_dict'].keys()
        for model in models:
            getattr(self, model).load_state_dict(checkpoint['model_state_dict'][model])
        optimizers = checkpoint['optimizer_state_dict'].keys()
        for optimizer in optimizers:
            getattr(self, optimizer).load_state_dict(checkpoint['optimizer_state_dict'][optimizer])
        losses = checkpoint['loss'].keys()
        for loss in losses:
            setattr(self, loss, checkpoint['loss'][loss])
            
        
    
    def load_state_old(self, fname, step = None):
        checkpoint = torch.load(fname)
        if step == None:
            self.step = checkpoint['step']
        else:
            step = step
        models = checkpoint['model_state_dict'].keys()
        for model in models:
            setattr(self, model,checkpoint['model_state_dict'][model])
        optimizers = checkpoint['optimizer_state_dict'].keys()
        for optimizer in optimizers:
            setattr(self, optimizer,checkpoint['optimizer_state_dict'][optimizer])
        losses = checkpoint['loss'].keys()
        for loss in losses:
            setattr(self, loss, checkpoint['loss'][loss])
        
    def fallback(self):
        for key in self.losses.keys():
            self.losses[key] = self.losses[key][:self.step]
            
        for key in self.lambdaLs.keys():
            self.lambdaLs[key] = self.lambdaLs[key][:self.step]
            
        for key in self.lrs.keys():
            self.lrs[key] = self.lrs[key][:self.step]
            
        for key in self.truthTables.keys():
            for ki in self.truthTables[key].keys():
                self.truthTables[key][ki] = self.truthTables[key][ki][:self.step]
        
    
    def failtest(self,failc = 1):
        failcondition = self.fliptest()+self.xLosstest()+self.cLosstest()
        if (failcondition) > failc:
            self.rand_lambdas()
            self.fix_lambdas()
            self.fix_lrs(np.random.rand(),dfac = self.lr_d_0, gfac = self.lr_g_0)
            if self.fallbackcount > 10:
                self.fallbackcount = 0
                self.maxstep = max(self.maxstep-100,0)
                print('reset maxstep!')
                
            if self.fallbackcount == 1:
                self.ll_0()
                
            self.save_lossdict(name=f'losses_fail{self.step}_{self.fallbackcount}')
            step = max(self.maxstep-200,0)
            step = np.round(step,-2)
            model_save_name = F'{self.save_run}/save/{self.run_name}_{step:06}.pt'
            
            self.load_state(model_save_name)
            self.fallback()
            self.fallbackcount += 1
            
            
            print(f'fallback to last state dict at step {self.step} - {self.fallbackcount}')
            
        else:
            self.fallbackcount = 0
            self.fix_lambdas()
            
            skewfac = 0.5*(self.tt_mean('tt_D_A') + self.tt_mean('tt_D_B'))
            self.fix_lrs(skewfac,dfac = self.lr_d_0, gfac = self.lr_g_0)
            print('no fail detected - continue training')
        return failcondition
        
        
        
        
        
        
        
        
        
        
        