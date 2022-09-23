import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.datasets.folder
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.models as models

from pathlib import Path

import advertorch
from advertorch import attacks
from advertorch.utils import predict_from_logits

sys.path.append("..")
from models.utils import Normalize, Permute
from dataset import ImageNet_png


parser = argparse.ArgumentParser()
parser.add_argument('--test_path', type=str,default="../output/",
                   help='path to test.')
parser.add_argument('--csv_path', type=str,default="../small_imagenet.csv",
                   help='path for csv to test.')
args=parser.parse_args()

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model = models.mnasnet1_0(pretrained=True).eval()
model = nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), model)
model.to(device)

batch_size=16

dataset = ImageNet_png(args.test_path, csv_name=args.csv_path)
loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=False)
    

count_non,count_tar=0,0

for batch_idx,data in enumerate(loader):
    img=data[0].to(device)
    true_label=(data[1]).to(device)
    target=(data[2]).to(device)
    output_cln=model(img)
    predict=torch.max(output_cln,1)[1]
    
    
    
    non_success =(true_label!=predict).cpu().sum().item()
    tar_success =(target==predict).cpu().sum().item()
    
    count_non+=non_success
    count_tar+=tar_success
    
    
print(count_non/len(dataset))

