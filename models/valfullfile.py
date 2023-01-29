import os
import torch
import torch.nn as nn
from models.networks import *
import numpy as np
import h5py
from models.data1channel import *
import time
# from dataset.CFibres1channel import *



        
# initialize network with weights from file_path
def load_model(file_path,model = UNetSim3d(num_out=1)):
    try:
        model = nn.DataParallel(model).to(device)
        state_dict = torch.load(file_path)['model_state_dict']['netG_A']
        model.load_state_dict(state_dict)
        model.eval()
    except:
        model = torch.load(file_path)['model_state_dict']['netG_A']
        model = nn.DataParallel(model).to(device)
        model.eval()

    model.eval()
    return model

# save new dataset to existing h5-file
def h5_add_dset(data,name='fake_B',set_name='[0,0,0]'):
    set_name=str(set_name)
    # print(set_name)
    path = "eval"
    if not os.path.exists(path):
        os.makedirs(path)
    save_name = f'{path}/temp_{name}.h5'
    data = data.squeeze()
    with h5py.File(save_name,'a') as f:
        f.create_dataset(set_name, shape = data.shape,dtype = 'float16',data = data)

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

def stitchh5(fname,shape,padding=10,wname = 'ev'):
    wname = f'eval/{name}.h5'
    with h5py.File(fname,'r') as rf, h5py.File(wname,'w') as wf:
        A = wf.create_dataset('data', dtype = np.float16,shape=shape)
        Am = wf.create_dataset('mask', dtype = np.int8,shape=shape)
        
        keys = [key for key in rf.keys()]
        lk = len(keys)
        print(f'found {lk} patches!')
        
        mask = np.squeeze(np.array(rf[keys[0]],np.float16))
        mask = padmask3(mask,padding)
        i=0
        for key in keys:
            pos = eval(key)
            B = np.squeeze(np.array(rf[key],np.float16)) * mask
            add3(A,B,pos)
            add3(Am,mask,pos)
            i+=1
            del B
            if i%10 == 0:
                print(f'key - {i}/{lk}', end='\r')
        Am[Am == 0] = 1
        for i in range(len(A)):
            A[i] =  np.array(A[i])/np.array(Am[i])
        del wf['mask']


def evalulate(datafile, modelfile, SR = 1, ev_name = 'ev'):
    st = time.time()
    print('start evaluating patches')
    print(time.asctime())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Dataset3dindex(dfile = datafile, settype = 'validate', dsize = (128,128,128), crop = 'origin')
    origins = dataset.orilist()
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16, shuffle=False)
    
    model = load_model(modelfile)

    with torch.no_grad():
        
        lk = len(loader)
        for k,(test_data,index) in enumerate(loader):

            out = model.forward(test_data)
            out = out.cpu()
            for l, idx in enumerate(index):
                data = out[l]
                h5_add_dset(data, name=ev_name, set_name=list(np.array(origins[idx])*SR))
            if k%10 == 0:
                print(f'enhancement - {k}/{lk}', end='\r')
        
    st1 = time.time()
    print(f'enhancement time = {st1-st} s')
    
    print('start stitching patches')
    padding = 10*SR
    stitchh5(f'eval/temp_{ev_name}.h5',shape = dataset.shapelist[0],padding = padding,wname = ev_name)

#if __name__ == '__main__':
    