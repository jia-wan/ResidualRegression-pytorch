import torch.nn as nn
import numpy as np
import torch
import math
import torch.nn.functional as F
from torchvision import models
from utils import save_net,load_net
from network import Conv2d



class Residual(nn.Module):
    '''
    given appearance estimation, deep features, and residual maps as input,
    output the final estimation
    '''
    def __init__(self, n=2, k_nn=3, bn=False):
        super(Residual, self).__init__()
        dim = 64
        self.r1 = nn.Sequential(Conv2d(n*64,32,3,NL='prelu',same_padding=True,bn=bn))
        self.r2 = nn.Sequential(Conv2d(n*64,16,5,NL='prelu',same_padding=True,bn=bn))
        self.r3 = nn.Sequential(Conv2d(n*64,8,7,NL='prelu',same_padding=True,bn=bn))
        self.residual_predict = nn.Sequential(
                                 Conv2d(56, 16, 7, NL='prelu', same_padding=True, bn=bn),
                                 Conv2d(16, 8, 5, NL='prelu', same_padding=True, bn=bn),
                                 Conv2d(8, 1, 3, NL='nrelu', same_padding=True, bn=bn))

        self.rm1 = nn.Sequential(Conv2d(k_nn,16,1,NL='relu',same_padding=True,bn=bn))
        self.rm2 = nn.Sequential(Conv2d(k_nn,8,3,NL='relu',same_padding=True,bn=bn))
        self.rm3 = nn.Sequential(Conv2d(k_nn,4,5,NL='relu',same_padding=True,bn=bn))
        self.residual_merge = nn.Sequential(Conv2d(28,1,3,NL='relu',same_padding=True,bn=bn))

        self.ram1 = nn.Sequential(Conv2d(2,16,1,NL='relu',same_padding=True,bn=bn))
        self.ram2 = nn.Sequential(Conv2d(2,8,3,NL='relu',same_padding=True,bn=bn))
        self.ram3 = nn.Sequential(Conv2d(2,4,5,NL='relu',same_padding=True,bn=bn))
        self.res_app_merge = nn.Sequential(Conv2d(28,1,3,NL='relu',same_padding=True,bn=bn))

        self._initialize_weights()

    def forward(self, features, app_prediction, support_gt):
        # pair test image and support images
        for i in range(len(features)-1):
            pair = torch.cat((features[0],features[i+1]),0)
            if i == 0:
                pairs = pair.unsqueeze(0)
            else:
                pairs = torch.cat((pairs, pair.unsqueeze(0)),0)
        # predict residual maps
        x1 = self.r1(pairs)
        x2 = self.r2(pairs)
        x3 = self.r3(pairs)
        pairs = self.residual_predict(torch.cat((x1,x2,x3),1))
        # calc residual based density estimation
        residual_predictions = pairs + support_gt.squeeze(0).unsqueeze(1)
        # merge residual mals
        n = len(residual_predictions)
        h = residual_predictions.shape[2]
        w = residual_predictions.shape[3]
        x1 = self.rm1(residual_predictions.view(1,n,h,w))
        x2 = self.rm2(residual_predictions.view(1,n,h,w))
        x3 = self.rm3(residual_predictions.view(1,n,h,w))
        final_residual_prediction = self.residual_merge(torch.cat((x1,x2,x3),1))
        # merge residual and appearance maps
        x1 = self.ram1(torch.cat((final_residual_prediction, app_prediction),1))
        x2 = self.ram2(torch.cat((final_residual_prediction, app_prediction),1))
        x3 = self.ram3(torch.cat((final_residual_prediction, app_prediction),1))
        final_prediction = self.res_app_merge(torch.cat((x1,x2,x3),1))
        #final_prediction = self.res_app_merge(torch.cat((app_prediction,final_residual_prediction),1))
        return final_prediction, residual_predictions, final_residual_prediction

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class CSRNet(nn.Module):
    def __init__(self, load_weights=False,norm=False,dropout=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512,256,128,64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 512,norm=norm,dilation = True, dropout=dropout)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self._initialize_weights()
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            fs = self.frontend.state_dict()
            ms = mod.state_dict()
            for key in fs:
                fs[key] = ms['features.'+key]
            self.frontend.load_state_dict(fs)
        else:
            print("Don't pre-train on ImageNet")

    def forward(self,x):
        x = self.frontend(x)
        x = self.backend(x)
        self.features = x
        x = self.output_layer(x)
        #x = F.interpolate(x, scale_factor=8)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels = 3,norm=False,dilation = False, dropout=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
