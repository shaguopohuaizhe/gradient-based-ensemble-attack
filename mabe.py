import sys, os
import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from dataset import ImageNet
from models.Denoise import resnet101_denoise,resnet152
from models.utils import Normalize,Permute
import cv2
from attacks import MABE


class Ensemble(nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = models

    def forward(self, x):
        logits = self.models[0](x)
        
        for i in range(len(self.models)-1):
            logits += self.models[i+1](x)
            
        
        logits_e = logits / len(self.models)

        return logits_e
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='./small_imagenet/', type=str, help='path to data')
    parser.add_argument('--output_dir', default='./mabe/', type=str, help='path to results')
    parser.add_argument('--csv', default='./small_imagenet.csv', type=str, help='path to csv')
    parser.add_argument('--batch_size', default=16, type=int, help='mini-batch size')
    parser.add_argument('--steps', default=20, type=int, help='iteration steps')
    parser.add_argument('--max_norm', default=16, type=float, help='Linf limit')
    args = parser.parse_args()


    # define models
    # resnet
    model_rn = models.resnet50(pretrained=True)
    model_rn = nn.Sequential(
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            model_rn
        )
    model_rn.cuda()
    model_rn.eval()
    # densenet
    model_dn = models.densenet121(pretrained=True)
    model_dn = nn.Sequential(
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            model_dn
        )
    model_dn.cuda()
    model_dn.eval()
    # vgg
    model_vgg = models.vgg19_bn(pretrained=True)
    model_vgg = nn.Sequential(
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            model_vgg
        )
    model_vgg.cuda()
    model_vgg.eval()
    # inception
    model_inc = models.inception_v3(pretrained=True)
    model_inc = nn.Sequential(
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            model_inc
        )
    model_inc.cuda()
    model_inc.eval()
    # denoise
    pretrained_model = resnet101_denoise()
    loaded_state_dict = torch.load(os.path.join('weight', 'Adv_Denoise_Resnext101.pytorch'))
    pretrained_model.load_state_dict(loaded_state_dict, strict=True)
    model_denoise = nn.Sequential(
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            Permute([2, 1, 0]),
            pretrained_model)
    model_denoise.cuda()
    model_denoise.eval()
    # adv
    pretrained_model = resnet152()
    loaded_state_dict = torch.load(os.path.join('weight', 'Adv_Resnet152.pytorch'))
    pretrained_model.load_state_dict(loaded_state_dict, strict=True)
    model_adv = nn.Sequential(
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            Permute([2, 1, 0]),
            pretrained_model)
    model_adv.cuda()
    model_adv.eval()
    
    #ensemble
    #model_ensemble = Ensemble([model_rn, model_dn, model_vgg, model_inc])
    #model_ensemble = [model_rn, model_dn, model_denoise, model_adv]
    model_ensemble = [model_rn, model_dn, model_vgg, model_inc]


    #define dataset
    dataset = ImageNet(args.csv)
    loader = torch.utils.data.DataLoader(dataset, 
                                         batch_size=args.batch_size, 
                                         shuffle=False)

    # set attacker
    adversary = MABE(
        model_ensemble, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=args.max_norm/255, nb_iter=args.steps, eps_iter=2*args.max_norm/args.steps/255, decay_factor=0.9, clip_min=0., clip_max=1., targeted=False)
    print("adversary set")
    
    for ind, (img, label_true, label_target, filenames) in enumerate(loader):
        # run attack
        adv = adversary.perturb(img.cuda(), label_true.cuda())
        print('finish batch {}'.format(ind))

        # save results
        for bind, filename in enumerate(filenames):
            out_img = adv[bind].detach().cpu().numpy()
            delta_img = np.abs(out_img - img[bind].numpy()) * 255.0


            out_img = np.transpose(out_img, axes=[1, 2, 0]) * 255.0
            out_img = out_img[:, :, ::-1]
            cleandir = os.path.split(filename)[-2]
            dirname = os.path.join(args.output_dir, os.path.split(cleandir)[-1])
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            out_filename = os.path.join(dirname, os.path.split(filename)[-1][:-5]+'.png')
            cv2.imwrite(out_filename, out_img)
        
    
           