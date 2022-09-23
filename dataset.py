import os
import random
import torch
import numpy as np
import glob
import pandas as pd
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImageNet(Dataset):
    def __init__(self, csv_name="/target.csv"):
        labels_dir = csv_name
        self.labels = pd.read_csv(labels_dir)

    def __len__(self):
        l = len(self.labels)
        return l

    def __getitem__(self, idx):
        filename = self.labels.at[idx, 'filename']
        in_img_t = cv2.imread(filename)[:, :, ::-1]
            
        in_img = np.transpose(in_img_t.astype(np.float32), axes=[2, 0, 1])
        img = in_img / 255.0

        label_true = self.labels.at[idx, 'label']
        label_target = self.labels.at[idx, 'target']
        
        return img, label_true, label_target, filename
        
class ImageNet_png(Dataset):
    def __init__(self, test, csv_name="/target.csv"):
        labels_dir = csv_name
        self.labels = pd.read_csv(labels_dir)
        self.test = test

    def __len__(self):
        l = len(self.labels)
        return l

    def __getitem__(self, idx):
        filename = self.labels.at[idx, 'filename']  
        
        cleandir = os.path.split(filename)[-2]
        cleanfile = os.path.split(filename)[-1][:-5]+'.png'
        dirname = os.path.split(cleandir)[-1]
        newdir = os.path.join(self.test,dirname)
        filename_t = os.path.join(newdir,cleanfile)
        in_img_t = cv2.imread(filename_t)[:, :, ::-1]
        in_img = np.transpose(in_img_t.astype(np.float32), axes=[2, 0, 1])
        img = in_img / 255.0

        label_true = self.labels.at[idx, 'label']
        label_target = self.labels.at[idx, 'target']
        	

        return img, label_true, label_target, filename
        
