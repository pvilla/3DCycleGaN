import os
import torch
import torch.nn as nn
from models.networks import *
import numpy as np
import h5py
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage.transform import rescale
from models.data1channel import *
import time
# from dataset.CFibres1channel import *


def orilist3Dfull(o_size=(200,200,200), patch_size=(64,64,64), stride=(1,1,1),ROI0=None,ROI1=None,ROI2=None):
    if type(o_size)==int:
        o_size = (o_size,o_size,o_size)
    
    if type(patch_size)==int:
        patch_size = (patch_size,patch_size,patch_size)
    
    if type(stride)==int:
        stride = (stride,stride,stride)
    
    if ROI0 == None:
        ROI0 = (0,o_size[0])
    if ROI1 == None:
        ROI1 = (0,o_size[1])
    if ROI2 == None:
        ROI2 = (0,o_size[2])
    
    i0s = np.arange(ROI0[0],ROI0[1]-(patch_size[0]-stride[0]),stride[0])
    i0s[-1] = ROI0[1] - patch_size[0]
    i1s = np.arange(ROI1[0],ROI1[1]-(patch_size[1]-stride[1]),stride[1])
    i1s[-1] = ROI1[1] - patch_size[1]
    i2s = np.arange(ROI2[0],ROI2[1]-(patch_size[2]-stride[2]),stride[2])
    i2s[-1] = ROI2[1] - patch_size[2]
    
    origins = []
    for k in i2s:
        for j in i1s:
            for i in i0s:
                origins.append([i,j,k])
    return origins
        

def load_model(file_path):
    model = nn.DataParallel(UNetSim3d(num_out=1)).to(device)
    try:
        state_dict = torch.load(file_path)['model_state_dict']['netG_A']
        model.load_state_dict(state_dict)
        model.eval()
    except:
        model = torch.load(file_path)['model_state_dict']['netG_A']
        model.eval()

    model.eval()
    return model



def h5_fake_B(data,name='fake_B',set_name='[0,0,0]'):
    set_name=str(set_name)
    # print(set_name)
    path = "eval"
    if not os.path.exists(path):
        os.makedirs(path)
    save_name = f'{path}/patches_{name}.h5'
    data = data.squeeze()
    with h5py.File(save_name,'a') as f:
        f.create_dataset(set_name, shape = data.shape,dtype = 'float16',data = data)


npath = '/data/staff/tomograms/users/johannes/2022_11_08_cycleGAN_HPFinder/results/HAD-con_lr034_128/save/HAD-con_lr034_128_011600.pt'
fval_A = '/data/staff/tomograms/users/johannes/2022_11_20_cycleGAN2_hpfinder/dataset/T700_GF4D_t05_pxs08_crop.json'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

origins = orilist3Dfull(o_size=(2015,1600,600), patch_size=(128,128,128), stride=(96,96,96))
#print(origins)
dataset = Dataset3dindex(dfile = fval_A, settype = 'validate', dsize = (128,128,128), crop = 'origin', origins=origins)
loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16, shuffle=False)


if __name__ == '__main__':
    st = time.time()
    print(time.asctime())
    model = load_model(npath)

    with torch.no_grad():
        
        lk = len(loader)
        for k,(test_data,index) in enumerate(loader):
            #print(k,index)
            # if k == 8:
            #     break
            out = model.forward(test_data)
            out = out.cpu()
            for l, idx in enumerate(index):
                data = out[l]
                h5_fake_B(data,name='HADCON_011700',set_name=list(np.array(origins[idx])*1))
            #print(out)
            if k%10 == 0:
                print(f'val - {k}/{lk}', end='\r')
        
    et = time.time()
    print(f'execution time = {et-st} s')
            