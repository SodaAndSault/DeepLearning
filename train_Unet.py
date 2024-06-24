import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable 
from torchvision.utils import save_image
from tqdm import tqdm
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import math
import torch, gc
import os
import random
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from complex_data import *
from Unet import *
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data_path = '../../ComplexNet/Brain/data/train/data'
train_label_path = '../../ComplexNet/Brain/data/train/label'
test_data_path = '../../ComplexNet/Brain/data/val/data'
test_label_path = '../../ComplexNet/Brain/data/val/label'

if __name__ == '__main__':
    
    train_data = get_training_set(train_data_path,train_label_path,2,10,0)
    test_data = get_training_set(test_data_path,test_label_path,2,10,0)
    train_iter = DataLoader(train_data,batch_size=2,shuffle=False)
    test_iter = DataLoader(test_data,batch_size=1,shuffle=False)
    
    save_weight_path = './param/Unet.pth'
    net = Unet()
    Loss = torch.nn.L1Loss()
    opt = optim.Adam(net.parameters(),lr = 0.00001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95,last_epoch=-1)
    
    pre_test_loss = 60
    loss_list = []
    ssim_list = []
    x = []
    epoch = 1
    for i in range(400):
        test_loss = 0
        epoch_loss = 0
        SSIM = 0
        PSNR = 0

        net.train()
        x.append(epoch)
        for data,label in tqdm(train_iter):
            net = net.cuda()
            data,label = data.cuda(),label.cuda() 
            out = net(data)
            train_loss = Loss(out,label)
            opt.zero_grad()
            train_loss.backward(retain_graph=True)
            opt.step() 
            epoch_loss = epoch_loss + train_loss.item()

        e_l = epoch_loss/len(train_iter)

        net.eval()
        with torch.no_grad():
            for index,(data,label) in enumerate(test_iter):
                data,label = data.cuda(),label.cuda()
                sim = 0
                ps=0
                out = net(data)
                test_loss += Loss(out,label).item()
        test_ls = test_loss/len(test_iter)
        if test_ls < pre_test_loss:
            torch.save(net.state_dict(),save_weight_path)
            pre_test_loss = test_ls

        print(f'{epoch}-train_loss ====>{e_l}')
        print(f'{epoch}-test_loss ====>{test_ls}')
        epoch = epoch+1