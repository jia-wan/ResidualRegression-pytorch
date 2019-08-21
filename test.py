# This is the testing script for "Residual Regression and Semantic Prior for crowd counting"
from model import Residual, CSRNet
import torch
import torch.nn as nn
from torchvision import datasets, transforms

import numpy as np
import argparse
import json
import cv2
import dataset

parser = argparse.ArgumentParser(description='Residual regression')
parser.add_argument('--test_json', metavar='TEST', type=str, default='part_A_test.json',
                    help='path to test json')
parser.add_argument('--n', metavar='Number', type=int, default=3,
                    help='number of support images')
parser.add_argument('--thr', metavar='THRESHOLD', type=float, default=0.7,
                    help='threshold for semantic prior')

transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def main():
    
    global args
    global support_imgs, support_gt, support_imgs_fea
    
    args = parser.parse_args()
    with open(args.test_json, 'r') as outfile:       
        val_list = json.load(outfile)
    
    residual = Residual(k_nn=args.n)
    residual = residual.cuda()

    counter = CSRNet()
    counter = counter.cuda()

    # load counter params
    checkpoint = torch.load('./saved_models/countercheckpoint.pth.tar')
    counter.load_state_dict(checkpoint['state_dict_model'])
    
    # load residual regressor params
    checkpoint = torch.load('./saved_models/residualcheckpoint.pth.tar')
    residual.load_state_dict(checkpoint['state_dict_res'])
    support_imgs = checkpoint['support_imgs'].cuda()
    support_gt = checkpoint['support_gt'].cuda()

    counter(support_imgs)
    support_imgs_fea = counter.features
            
    app, res, final = validate(val_list, counter, residual)
        
    
def validate(val_list, counter, residual):
    print ('begin test')
    test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   transform=transform),
    batch_size=1)    
    
    app_mae = 0
    res_mae = 0
    final_mae = 0
    
    i = -1
    for data in (test_loader):
        i += 1
        img, target, smap = data
        img = img.cuda()
        output = counter(img)
        output_fea = counter.features

        n_h = int(output_fea.shape[2]/support_imgs_fea.shape[2])
        n_w = int(output_fea.shape[3]/support_imgs_fea.shape[3])
        fea = torch.cat((output_fea, support_imgs_fea.repeat(1,1,n_h,n_w)),0)
        # CSRNet is used to extract image features
        final_output, _, final_residual = residual(fea, output, support_gt.repeat(1,1,n_h,n_w))

        # semantic map
        smap = smap.type(torch.cuda.FloatTensor).unsqueeze(0)
        smap[smap > 0] = 1
        smap[smap <= 0] = args.thr
        
        final_output = torch.mul(final_output, smap)

        app_mae += abs(output[0].data.sum()-target.sum().type(torch.FloatTensor).cuda())
        res_mae += abs(final_residual.data.sum()-target.sum().type(torch.FloatTensor).cuda())
        final_mae += abs(final_output.data.sum()-target.sum().type(torch.FloatTensor).cuda())
        
    app_mae = app_mae/len(test_loader)
    res_mae = res_mae/len(test_loader)    
    final_mae = final_mae/len(test_loader)
    print(' * app MAE {mae:.3f}  * residual MAE {res_mae:.3f}  * final MAE {final_mae:.3f}'
              .format(mae=app_mae, res_mae=res_mae, final_mae=final_mae))

    return app_mae, res_mae, final_mae
        
    
if __name__ == '__main__':
    main()        
