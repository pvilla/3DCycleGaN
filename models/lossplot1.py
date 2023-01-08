#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 18:16:17 2022

@author: jestubbe
"""
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

lossfile = '/data/staff/tomograms/users/johannes/2022_09_26-cycleSegLL/results/A50_B50_10_GA30_GB30_lrG1e-04_lrD1e-04_seg2_semi_bin-dice/losses.json'
lossfile = '/data/staff/tomograms/users/johannes/2022_09_26-cycleSegLL/results/A50_B50_10_0_GA30_GB30_lrG1e-04_lrD1e-03_seg2_semi_bin-dice/losses.json'
def fitplot(x,y):
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x,p(x),c='grey')
    return z

def lossplot200(d,last=-1,name = 'loss',path = ''):
    plt.figure(figsize=(20,10))
    d = d[last-200:last]
    N=15
    mean = np.mean(d)
    std = np.std(d)
    mi = np.max([mean - 3*std,0])
    ma = mean + 3*std
    d_rmean = np.convolve(d, np.ones(N)/N, mode='valid')
    plt.plot([], [], ' ', label=f'mean: {mean:.4f}')
    plt.plot([], [], ' ', label=f'std: {std:.4f}')
    x = np.arange(-len(d),0)
    x_rmean = np.arange(-200+(N/2),-200+(N/2)+len(d_rmean))
    f_full = fitplot(x,d)
    f_0 = fitplot(x[:100],d[:100])
    f_1 = fitplot(x[100:],d[100:])
    plt.plot([], [], ' ', label=f'grad_full: {100*f_full[0]:.4f} %')
    plt.plot([], [], ' ', label=f'grad_0: {100*f_0[0]:.4f} %')
    plt.plot([], [], ' ', label=f'grad_1: {100*f_1[0]:.4f} %')

    plt.ylim(mi,ma)
    plt.plot(x,d,lw=.5)
    plt.plot(x_rmean, d_rmean,lw=2)
    plt.xlabel('steps')
    plt.legend(loc='lower left',title = f'{name} stats:')
    plt.savefig(f'{path}/loss200_{name}.png')

def lossplotFull(d,esteps = 500,name = 'loss',path = ''):
    plt.figure(figsize=(20,10))
    d = d
    N=len(d)//12
    mean = np.mean(d)
    std = np.std(d)
    mi = np.max([mean - 3*std,0])
    ma = mean + 3*std
    d_rmean = np.convolve(d, np.ones(N)/N, mode='valid')
    plt.plot([], [], ' ', label=f'mean: {mean:.4f}')
    plt.plot([], [], ' ', label=f'std: {std:.4f}')
    x = np.linspace(0,len(d)/esteps,len(d))
    x_rmean = x[N//2:(N//2)+len(d_rmean)]
    f = fitplot(x[150:],d[150:])
    plt.plot([], [], ' ', label=f'grad_full: {100*f[0]:.2f} %')

    plt.ylim(mi,ma)
    plt.plot(x,d,lw=.5)
    plt.plot(x_rmean, d_rmean,lw=2)
    plt.xlabel('epochs')
    plt.legend(loc='lower left',title = f'{name} stats:')
    plt.savefig(f'{path}/lossf_{name}.png')

def ttplotFull(ttsk,esteps = 500,name = 'tt',path = ''):
    keys_1 = ['TP','TN','FP','FN']
    s = 0
    for k1 in keys_1:
        try:
            s += ttsk[k1][0]
        except:
            pass
    plt.figure(figsize=(20,10))
    a = 0
    a_rmean = 0
    for i,k1 in enumerate(keys_1):
            d = np.array(ttsk[k1])/s
            N=len(d)//12
            x = np.linspace(0,len(d)/esteps,len(d))
            d_rmean = np.convolve(d, np.ones(N)/N, mode='valid')
            x_rmean = x[N//2:(N//2)+len(d_rmean)]
            plt.plot(x,d,lw=.5,c=f'C{i}',zorder=.1,ls='dotted')
            plt.plot(x_rmean,d_rmean,lw=2,c=f'C{i}',label=k1,zorder=.9)
            if k1[0] == 'T':
                a += d 
                a_rmean += d_rmean
    plt.plot(x,a,lw=.5,c=f'C{i+1}',zorder=1,ls='dotted')
    plt.plot(x_rmean,a_rmean,lw=2,c='black',label='Accuracy',zorder=.95)
    
    mean = np.mean(a)
    std = np.std(a)
    plt.plot([], [], ' ', label=f'mean_A: {mean:.4f}')
    plt.plot([], [], ' ', label=f'std_A: {std:.4f}')
    # f = fitplot(x[150:],a[150:])
    # plt.plot([], [], ' ', label=f'grad_A: {100*f[0]:.2f} %')
    plt.ylim(0,1)
    plt.xlabel('epochs')
    plt.legend(loc = 'upper left', title = f'{name} stats:')
    plt.savefig(f'{path}/ttf_{name}.png')
    
def ttplot200(ttsk,last = -1,name = 'tt',path = ''):
    keys_1 = ['TP','TN','FP','FN']
    s = 0
    for k1 in keys_1:
        try:
            s += ttsk[k1][0]
        except:
            pass
    plt.figure(figsize=(20,10))
    a = 0
    a_rmean = 0
    for i,k1 in enumerate(keys_1):
            d = np.array(ttsk[k1][last-200:last])/s
            N=10
            x = np.arange(-len(d),0)
            d_rmean = np.convolve(d, np.ones(N)/N, mode='valid')
            x_rmean = x[N//2:(N//2)+len(d_rmean)]
            plt.plot(x,d,lw=.5,c=f'C{i}',zorder=.1,ls='dotted')
            plt.plot(x_rmean,d_rmean,lw=2,c=f'C{i}',label=k1,zorder=.9)
            if k1[0] == 'T':
                a += d 
                a_rmean += d_rmean
    plt.plot(x,a,lw=.5,c=f'C{i+1}',zorder=1,ls='dotted')
    plt.plot(x_rmean,a_rmean,lw=2,c='black',label='Accuracy',zorder=.95)
    
    mean = np.mean(a)
    std = np.std(a)
    plt.plot([], [], ' ', label=f'mean_A: {mean:.4f}')
    plt.plot([], [], ' ', label=f'std_A: {std:.4f}')
    f_full = fitplot(x,a)
    f_0 = fitplot(x[:100],a[:100])
    f_1 = fitplot(x[100:],a[100:])
    plt.plot([], [], ' ', label=f'grad_full: {100*f_full[0]:.4f} %')
    plt.plot([], [], ' ', label=f'grad_0: {100*f_0[0]:.4f} %')
    plt.plot([], [], ' ', label=f'grad_1: {100*f_1[0]:.4f} %')
    plt.ylim(0,1)
    plt.xlabel('steps')
    plt.legend(loc = 'upper left', title = f'{name} stats:')
    plt.savefig(f'{path}/tt200_{name}.png')


def plotmetrics(lossfile):
    
    with open(lossfile) as jf:
        data = json.load(jf)
    path,file = os.path.split(lossfile)
    losses = data['losses']
    lambdas = data['lambdas']
    tts = data['tts']
    losskeys = losses.keys()
    for key in losskeys:
        try:
            d=losses[key]
            lossplotFull(d,1,key,path)
            lossplot200(d,-1,key,path)
        except:
            print(f'{key} failed')
        
    
    ttkeys = tts.keys()
    for key in ttkeys:
        # try:
            ttsk = tts[key]
            ttplotFull(ttsk,1,key,path)
            ttplot200(ttsk,-1,key,path)
            plt.close('all')
        # except:
        #     print(f'{key} failed')
        #     plt.close('all')
      
        
if __name__ == '__main__':
    lossfiles = [
        '/data/staff/tomograms/users/johannes/2022_11_08_cycleGAN_HPFinder/results/A10_B10_GA10_GB10_lrG1e-05_lrD1e-05_hpfinder/losses_fail99.json'
        ]
    
    for lf in lossfiles:
        plotmetrics(lf)
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    