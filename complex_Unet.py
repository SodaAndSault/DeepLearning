from complex_layers import *
from torch import nn
from torch.nn import functional as F
import torch 

class Conv_Bloc(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Conv_Bloc,self).__init__()
        self.layer1 = nn.Sequential(
            ComplexConv2d(in_channel,out_channel,3,1,1),
            ComplexBatchNorm2d(out_channel),)
        self.layer2 = nn.Sequential(
            ComplexConv2d(out_channel,out_channel,3,1,1),
            ComplexBatchNorm2d(out_channel),
        )  
            
    
    def forward(self,x):
        return complex_relu(self.layer2(complex_relu(self.layer1(x))))



class DownSample(nn.Module):
    def __init__(self,channel):
        super(DownSample,self).__init__()
        self.layer = nn.Sequential(
            ComplexConv2d(channel,channel,3,2,1),
            ComplexBatchNorm2d(channel),
        )
    def forward(self,x):
        return complex_relu(self.layer(x))

class UpSample(nn.Module):
    def __init__(self,channel):
        super(UpSample,self).__init__()
        self.layer = ComplexConv2d(channel,int(channel/2),1,1)
    def forward(self,x,feature_map):
        up = complex_upsample(x,scale_factor=(2,2),mode='nearest')
        out = self.layer(up)
        return torch.cat([out,feature_map],dim=1)


class Unet_IQ(nn.Module):
    def __init__(self):
        super(Unet_IQ,self).__init__()
        self.c1 = Conv_Bloc(1,64)
        self.d1 = DownSample(64)
        self.c2 = Conv_Bloc(64,128)
        self.d2 = DownSample(128)
        self.c3 = Conv_Bloc(128,256) 
        self.d3 = DownSample(256)
        self.c4 = Conv_Bloc(256,512)

        self.u1 = UpSample(512)
        self.c5 = Conv_Bloc(512,256)
        self.u2 = UpSample(256)
        self.c6 = Conv_Bloc(256,128)
        self.u3 = UpSample(128)
        self.c7 = Conv_Bloc(128,64)

        self.out = ComplexConv2d(64,1,3,1,1)

    def forward(self,x):
        R1 = self.c1(x)
        #print(R1.shape)
        R2 = self.c2(self.d1(R1))
        #print(R2.shape)
        R3 = self.c3(self.d2(R2))
        #print(R3.shape)
        R4 = self.c4(self.d3(R3))
        #print(R4.shape)
        
        o1 = self.c5(self.u1(R4,R3))
        #print(o1.shape)
        o2 = self.c6(self.u2(o1,R2))
        #print(o2.shape)
        o3 = self.c7(self.u3(o2,R1))
        # print(o3.shape)

        return R1,R2,R3,R4,o1,o2,o3,self.out(o3)

