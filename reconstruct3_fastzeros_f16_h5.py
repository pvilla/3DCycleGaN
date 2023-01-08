#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 23:48:35 2022

@author: jestubbe
"""
import numpy as np
import h5py
import time

def merge3(array_A,array_B,pos_B):
    shape_A = np.array(np.shape(array_A))
    shape_B = np.array(np.shape(array_B))
    shape_B_pos = shape_B + np.array(pos_B)
    shape_out = np.maximum(shape_A,shape_B_pos)
    if not np.array_equal(shape_out, shape_A):
        z = np.zeros_like(array_B,shape=tuple(shape_out))
        A0_1, A1_1, A2_1 = shape_A
        z[0:A0_1,0:A1_1,0:A2_1] += array_A
        array_A = z
    B0_1, B1_1, B2_1 = shape_B_pos
    B0_0, B1_0, B2_0 = pos_B
    array_A[B0_0:B0_1,B1_0:B1_1,B2_0:B2_1] += array_B
    return array_A


def add3(array_A,array_B,pos_B):
    shape_B = np.array(np.shape(array_B))
    shape_B_pos = shape_B + np.array(pos_B)
    B0_1, B1_1, B2_1 = shape_B_pos
    B0_0, B1_0, B2_0 = pos_B
    array_A[B0_0:B0_1,B1_0:B1_1,B2_0:B2_1] += array_B

def padmask3(a,padding=10):
    p = padding
    mask = np.zeros_like(a,dtype=np.int8)
    mask[p:-p,p:-p,p:-p] = 1
    return mask

def stitch(fname):
    A = np.array([[[0.]]],np.float16)
    Am = np.array([[[0]]],np.int8)
    with h5py.File(fname,'r') as f:            
        keys = [key for key in f.keys()]
        lk = len(keys)
        print(f'found {lk} patches!')
        mask = np.squeeze(np.array(f[keys[0]],np.float16))
        mask = padmask3(mask,10)
        
        i=0
        for key in keys:
            pos = eval(key)
            #print(pos)
            B = np.squeeze(np.array(f[key],np.float16)) * mask
            A = merge3(A,B,pos)
            Am = merge3(Am,mask,pos)
            i+=1
            del B
            if i%10==0:
                print(f'key - {i}/{lk}', end='\r')
            # if i>50:
            #     break
    return(A,Am)

def stitchh5(fname):
    shape = (2016,1601,601)
    wname = fname[:-3] + '_patched_h5.h5'
    with h5py.File(fname,'r') as rf, h5py.File(wname,'w') as wf:
        A = wf.create_dataset('patches', dtype = np.float16,shape=shape)
        Am = wf.create_dataset('mask', dtype = np.int8,shape=shape)
        
        keys = [key for key in rf.keys()]
        lk = len(keys)
        print(f'found {lk} patches!')
        
        mask = np.squeeze(np.array(rf[keys[0]],np.float16))
        mask = padmask3(mask,10)
        i=0
        for key in keys:
            pos = eval(key)
            B = np.squeeze(np.array(rf[key],np.float16)) * mask
            add3(A,B,pos)
            add3(Am,mask,pos)
            i+=1
            del B
            if i%10==0:
                print(f'key - {i}/{lk}', end='\r')
        Am[Am == 0] = 1
        An = np.array(A)/np.array(Am)
        wf.create_dataset('normalized', dtype = 'float16', data = An)

fname = '/data/staff/tomograms/users/johannes/2022_11_08_cycleGAN_HPFinder/eval/patches_HADCON_011700.h5'

if __name__ == '__main__':
    print(time.asctime())
    st = time.time()
    stitchh5(fname)
    et = time.time()
    print(f'execution time = {et-st} s')
        
        
        
        
        
        
        