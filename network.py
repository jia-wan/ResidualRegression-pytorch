import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True) if bn else None
        if NL == 'relu':
            self.relu = nn.ReLU(inplace=True)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        elif NL == 'lrelu':
            self.relu = nn.LeakyReLU(0.2, True)
        else:
            self.relu = None	

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def save_net(fname, net):
    import h5py
    with h5py.File(fname, mode='w') as h5f:
        for k, v in net.state_dict().items():
            #if 'module' in k:
            #    k = k.replace('.module','.')
            h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    import h5py
    with h5py.File(fname, mode='r') as h5f:
        for k, v in net.state_dict().items():        
            #print(k)
            param = torch.from_numpy(np.asarray(h5f[k]))         
            v.copy_(param)


def np_to_variable(x, is_cuda=True, is_training=False, dtype=torch.FloatTensor):
    if is_training:
        v = Variable(torch.from_numpy(x).type(dtype))
    else:
        v = Variable(torch.from_numpy(x).type(dtype), requires_grad = False, volatile = True)
    if is_cuda:
        v = v.cuda()
    return v

def to_variable(x, is_cuda=True, is_training=False):
    if is_training:
        v = Variable(x)
    else:
        v = Variable(x, requires_grad = False, volatile = True)
    if is_cuda:
        v = v.cuda()
    return v
    

def set_trainable(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):                
                #print torch.sum(m.weight)
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.PReLU):
                m.weight.data.fill_(0.001)


# put i-th map to the top
def to_top(data, i):
    temp = data
    if(i == 0):                                                                                                                 
        return temp
    temp[0,:,:,:,] = data[i,:,:,:]
    temp[i,:,:,:,] = data[0,:,:,:]
    return temp

