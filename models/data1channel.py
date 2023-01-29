import h5py
import torch
from torch.utils import data
import numpy as np
import json


#### image manipulations. can be called in the Dataset3Dxxx
def standardize(x,mean,std):
    return((x-mean)/std)

def autostandardize(x):
    mean = np.mean(x)
    std = np.std(x)
    return((x-mean)/std)

def randomflip(x):
    for n in range(3):
        if np.random.randint(0,2) == 1:
            x = np.flip(x,-n).copy()
    return x

def read_json(file):
    try:
        with open(file) as jf:
            return json.load(jf)
    except:
        d = {"validate":[{"path": file, "dset": "data"}]}
        return d

def centercrop(x,size):             # crop the image to a central square
    off_t = (np.shape(x)[-2]-size)//2
    off_l = (np.shape(x)[-1]-size)//2
    return(x[:,off_t:off_t+size,off_l:off_l+size])

def randomcrop(x,size):             # crop a random part of the image to a square
    off_t = np.random.randint(1, (np.shape(x)[-2]-size)-1)
    off_l = np.random.randint(1, (np.shape(x)[-1]-size)-1)
    return(x[:,off_t:off_t+size,off_l:off_l+size])

def centercrop3d(x,size):             # crop the image to a central square
    off_3 = (np.shape(x)[-3]-size)//2
    off_2 = (np.shape(x)[-2]-size)//2
    off_1 = (np.shape(x)[-1]-size)//2
    return(x[off_3:off_3+size,off_2:off_2+size,off_1:off_1+size])

def randomcrop3d(x,size):             # crop a random part of the image to a square
    off_3 = np.random.randint(1, (np.shape(x)[-3]-size)-1)
    off_2 = np.random.randint(1, (np.shape(x)[-2]-size)-1)
    off_1 = np.random.randint(1, (np.shape(x)[-1]-size)-1)
    return(x[off_3:off_3+size,off_2:off_2+size,off_1:off_1+size])
###

def h5listshape(flist,dset):          #returns the amount of items from specific datasets in multiple h5 files.
    ls = []
    for i in range(len(flist)):
        with h5py.File(flist[i],'r') as f:
            ls.append(np.shape(f[dset[i]]))
    ls = np.array(ls)
    return(ls)

def h5SizeFetch(fname,dset,size,origin):          # fetches an element from an h5 file
    size = np.array(size,dtype=int)
    origin = np.array(origin,dtype=int)
    stop = origin + size
    with h5py.File(fname,'r') as f:
        x = np.array(f[dset][...,origin[0]:stop[0],origin[1]:stop[1],origin[2]:stop[2]], dtype=np.float32)
        x[np.isfinite(x) == 0] = 0
    return(x)

def json_to_list(data):         #read the json files that define the h5 datasets
    path = []
    dset = []
    ROI0 = []
    ROI1 = []
    ROI2 = []
    for element in data:
        path.append(element['path'])
        if 'dset' in element:
            dset.append(element['dset'])
        else:
            dset.append('data')
            
        if 'ROI0' in element:
            ROI0.append(element['ROI0'])
        else:
            ROI0.append([0,-1])
            
        if 'ROI1' in element:
            ROI1.append(element['ROI1'])
        else:
            ROI1.append([0,-1])
            
        if 'ROI2' in element:
            ROI2.append(element['ROI2'])
        else:
            ROI2.append([0,-1])
            
    path = np.array(path)
    dset = np.array(dset)
    ROI0 = np.array(ROI0,dtype=np.int)
    ROI1 = np.array(ROI1,dtype=np.int)
    ROI2 = np.array(ROI2,dtype=np.int)
        
    return(path,dset,ROI0,ROI1,ROI2)
        
def fix_ROI_list(rois,shapes):
    for im in range(len(rois)):
        i = -1 - im
        if rois[i][0] < 0:
            rois[i][0] = shapes[i]+rois[i][0]
        
        if rois[i][1] < 0:
            rois[i][1] = shapes[i]+rois[i][1]
    return rois

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

class Dataset3dsingle(data.Dataset): #h5 dataloader. dfile - json dict with filenames. settype - 'train' or 'validate'. crop - 'random', 'origin',  or tuple interpreted as origin. dlen - sample number. origins - list if crop='origins'. manipulate - data-augmentation.
    def __init__(self, dfile, settype = 'train', dsize = (128,128,128), crop = 'random', dlen = 10000, origins = None, manipulate = None):
        if settype == 'validate':
            alttype = 'train'
        else:
            settype = 'train'
            alttype = 'validate'
        
        d = read_json(dfile)
        if settype in d:
            self.d_path, self.d_dset, self.d_ROI0, self.d_ROI1, self.d_ROI2 = json_to_list(d[settype])
        else:
            self.d_path, self.d_dset, self.d_ROI0, self.d_ROI1, self.d_ROI2 = json_to_list(d[alttype])
        
        if type(crop) == tuple:
            self.origins = [crop]
            crop = 'origin'
        self.crop = crop
        self.origins = origins
        self.dlen = dlen
        self.manipulate = manipulate
        self.dsize = np.array(dsize)
        self.shapelist = h5listshape(self.d_path,self.d_dset)
        
        self.d_ROI0 = fix_ROI_list(self.d_ROI0, self.shapelist[:,-3])
        self.d_ROI1 = fix_ROI_list(self.d_ROI1, self.shapelist[:,-2])
        self.d_ROI2 = fix_ROI_list(self.d_ROI2, self.shapelist[:,-1])

    def __len__(self):
        if self.crop == 'random':
            return self.dlen
        if self.crop == 'origin':
            return len(self.origins)
        
        
        
    def __getitem__(self, index):
        if self.crop == 'random':
            fi = np.random.randint(0,len(self.d_path))    #file index
            c0 = np.random.randint(self.d_ROI0[fi][0],self.d_ROI0[fi][1]-self.dsize[-3])
            c1 = np.random.randint(self.d_ROI1[fi][0],self.d_ROI1[fi][1]-self.dsize[-2])
            c2 = np.random.randint(self.d_ROI2[fi][0],self.d_ROI2[fi][1]-self.dsize[-1])
            origin = (c0,c1,c2)
                        
        if self.crop == 'origin':
            fi = 0    #file index
            origin = self.origins[index]
        
        A = h5SizeFetch(self.d_path[fi], self.d_dset[0], self.dsize, origin)
        
        if self.manipulate != None:
            A = self.manipulate(A)
            
        while len(np.shape(A))<4:
            A = np.expand_dims(A, 0)
        
        A = torch.tensor(A,dtype=torch.float32)
        return A
    
class Dataset3dindex(data.Dataset):             #returns Dataset and index. Needed for multibatch sliding window
    def __init__(self, dfile, settype = 'train', dsize = (128,128,128), crop = 'random', dlen = 10000, origins = None):
        if settype == 'train':
            alttype = 'validate'
        elif settype == 'validate':
            alttype = 'train'
        else:
            # print('settype not recognized. Using "train" instead')
            settype = 'train'
            alttype = 'validate'
        
        d = read_json(dfile)
        if settype in d:
            self.d_path, self.d_dset, self.d_ROI0, self.d_ROI1, self.d_ROI2 = json_to_list(d[settype])
        else:
            self.d_path, self.d_dset, self.d_ROI0, self.d_ROI1, self.d_ROI2 = json_to_list(d[alttype])
        
        if type(crop) == tuple:
            self.origins = [crop]
            crop = 'origin'
        self.crop = crop
        self.origins = origins
        self.dlen = dlen
        self.dsize = np.array(dsize)
        self.shapelist = h5listshape(self.d_path,self.d_dset)
        
        self.d_ROI0 = fix_ROI_list(self.d_ROI0, self.shapelist[:,0])
        self.d_ROI1 = fix_ROI_list(self.d_ROI1, self.shapelist[:,1])
        self.d_ROI2 = fix_ROI_list(self.d_ROI2, self.shapelist[:,2])

    def __len__(self):
        if self.crop == 'random':
            return self.dlen
        if self.crop == 'origin':
            return len(self.origins)
    
    def orilist(self):      #generate list of origins for sliding windows
        stride = self.dsize - 24
        oris = []
        for i in range(len(self.shapelist)):
            ori = orilist3Dfull(self.shapelist[i],patch_size=self.dsize,stride=stride,ROI0=self.d_ROI0[i],ROI1=self.d_ROI1[i],ROI2=self.d_ROI2[i])
            oris.append(ori)
        self.origins = oris[0]
        return oris[0]
    
    def __getitem__(self, index):
        
        if self.crop == 'random':
            fi = np.random.randint(0,len(self.d_path))    #file index
            c0 = np.random.randint(self.d_ROI0[fi][0],self.d_ROI0[fi][1]-self.dsize[0])
            c1 = np.random.randint(self.d_ROI1[fi][0],self.d_ROI1[fi][1]-self.dsize[1])
            c2 = np.random.randint(self.d_ROI2[fi][0],self.d_ROI2[fi][1]-self.dsize[2])
            origin = (c0,c1,c2)
                        
        if self.crop == 'origin':
            fi = 0    #file index; fix this to include all files!!!!
            origin = self.origins[index]
        
        A = h5SizeFetch(self.d_path[fi], self.d_dset[0], self.dsize, origin)
        
        A = np.expand_dims(A, 0)
        
        A = torch.tensor(A,dtype=torch.float32)
            
        return A, index
    

class Dataset3d(data.Dataset):
    def __init__(self, dfile_A, dfile_B, settype = 'train', dsize_A = (128,128,128), dsize_B = (128,128,128), crop = 'random', dlen = 10000, manipulate = randomflip):
        self.Ds_A = Dataset3dsingle(dfile_A, settype = settype, dsize = dsize_A, crop = crop, dlen = dlen, origins = None, manipulate = manipulate)
        self.Ds_B = Dataset3dsingle(dfile_B, settype = settype, dsize = dsize_B, crop = crop, dlen = dlen, origins = None, manipulate = manipulate)
     

    def __len__(self):
        return self.Ds_A.__len__()
        
    def __getitem__(self, index):
        A = self.Ds_A.__getitem__(index)
        B = self.Ds_B.__getitem__(index)
        return (A, B)
    
