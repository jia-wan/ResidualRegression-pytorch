import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from image import load_data
import torchvision.transforms.functional as F

class listDataset(Dataset):
    def __init__(self, root, transform, num_workers=4):
        self.lines = root
        self.nSamples = len(root)
        self.transform = transform
        
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 

        img_path = self.lines[index]

        img,target,smap = load_data(img_path)        
        img = self.transform(img)

        return img,target,smap

