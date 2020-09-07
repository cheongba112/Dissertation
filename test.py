import os, random, csv, time, re, shutil
import numpy as np
# from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from options import *
from misc import *
from get_dataset import *
from models import *


res_pth = './test_result/'

dataset = get_dataset(opt.dataroot + '_test')

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

netE = Encoder().to(device)
netG = Generator().to(device)
netR = AgeRegressor().to(device)

L1 = nn.L1Loss().to(device)
CE = nn.CrossEntropyLoss().to(device)
MSE = nn.MSELoss().to(device)

def main():
    if not os.path.exists(res_pth):
        os.makedirs(res_pth)

    netE.load_state_dict(torch.load(opt.netE_path))
    netE.eval()
    netG.load_state_dict(torch.load(opt.netG_path))
    netG.eval()
    netR.load_state_dict(torch.load(opt.netR_path))
    netR.eval()
    
    err = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    num = [0,  0,  0,  0,  0,  0,  0,  0,  0,  0 ]
    
    for i, (src_img, src_age) in enumerate(dataloader):

        # prepare batch data
        src_img = src_img.to(device)
        src_age = src_age.to(device)

        loss_values = torch.zeros(1, 10).to(device)

        for syn_age in range(10):
            syn_age_onehot = torch.zeros(1, 10).to(device)
            syn_age_onehot[0][syn_age] = 1.

            syn_vec = netE(src_img, syn_age_onehot)
            syn_img = netG(syn_vec, syn_age_onehot)

            loss_values[0][syn_age] = L1(syn_img, src_img)

        pred_age = netR(loss_values).view(-1)
        loss_r = MSE(pred_age.float(), src_age.float())
        
        if   src_age <= 5:  err[0] += loss_r.tolist(); num[0] += 1 
        elif src_age <= 10: err[1] += loss_r.tolist(); num[1] += 1 
        elif src_age <= 15: err[2] += loss_r.tolist(); num[2] += 1 
        elif src_age <= 20: err[3] += loss_r.tolist(); num[3] += 1 
        elif src_age <= 30: err[4] += loss_r.tolist(); num[4] += 1 
        elif src_age <= 40: err[5] += loss_r.tolist(); num[5] += 1 
        elif src_age <= 50: err[6] += loss_r.tolist(); num[6] += 1 
        elif src_age <= 60: err[7] += loss_r.tolist(); num[7] += 1 
        elif src_age <= 70: err[8] += loss_r.tolist(); num[8] += 1 
        else:               err[9] += loss_r.tolist(); num[9] += 1 
    
    for i in range(10):
        print('group %d L1 loss: %f'% (i, err[i] / num[i]))

    print(sum(err) / sum(num))
        
if __name__ == '__main__':
    main()
