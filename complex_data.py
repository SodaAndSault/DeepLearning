
import torch
import numpy as np
import torch.utils.data as data
from os import listdir
import os
import scipy.io as io
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import hdf5storage
from torch.utils.data import DataLoader
import torch.nn.functional as F

class get_training_set(data.Dataset):
    def __init__(self,data_path,label_path,pad_h,pad_w,if_complex):
        super(get_training_set,self).__init__()
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.data_path = data_path
        self.label_path = label_path
        self.image_list = os.listdir(data_path)
        self.image_list.sort(key=lambda x:int(x[:-4]))
        self.if_complex = if_complex

    def __getitem__(self,index):
        data_path = os.path.join(self.data_path,self.image_list[index])
        label_path = os.path.join(self.label_path,self.image_list[index])
        input_ = torch.tensor(io.loadmat(data_path)['data'],dtype=torch.complex64,requires_grad=True).permute(2,0,1)
        input = F.pad(input_,(0,self.pad_w,0,self.pad_h))
        target_ = torch.tensor(io.loadmat(label_path)['target'],dtype=torch.complex64,requires_grad=True).permute(2,0,1)
        target = F.pad(target_,(0,self.pad_w,0,self.pad_h))
        data = torch.unsqueeze(input.reshape(input.size(0),input.size(1)*input.size(2)),0)
        label = torch.unsqueeze(target.reshape(target.size(0),target.size(1)*target.size(2)),0)
        if(self.if_complex==1):
            return data,label
        else:
            return torch.sqrt(data.real ** 2 + data.imag ** 2),torch.sqrt(label.real ** 2 + label.imag ** 2)
    
    def __len__(self):
        return len(self.image_list)


class get_test_set(data.Dataset):
    def __init__(self,data_path,pad_h,pad_w,if_complex):
        super(get_test_set,self).__init__()
        self.data_path = data_path
        self.image_list = os.listdir(data_path)
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.image_list.sort(key=lambda x:int(x[:-4]))
        self.if_complex = if_complex

    def __getitem__(self,index):
        data_path = os.path.join(self.data_path,self.image_list[index])
        input_ = torch.tensor(io.loadmat(data_path)['data'],dtype=torch.complex64,requires_grad=True).permute(2,0,1)
        input = F.pad(input_,(0,self.pad_w,0,self.pad_h))
        data = torch.unsqueeze(input.reshape(input.size(0),input.size(1)*input.size(2)),0)
        if(self.if_complex==1):
            return data
        else:
            return torch.sqrt(data.real ** 2 + data.imag ** 2)
    
    def __len__(self):
        return len(self.image_list)