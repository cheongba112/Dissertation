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

# --------------------------------------------------
# main

res_pth = './regression_result/'

dataset = get_dataset(opt.dataroot)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

netE = Encoder().to(device)
netG = Generator().to(device)
netR = AgeRegressor().to(device)

L1 = nn.L1Loss().to(device)
CE = nn.CrossEntropyLoss().to(device)

optimR = optim.Adam(netR.parameters(), lr=0.0002, betas=(0.5, 0.999))

def main():
    if not os.path.exists(res_pth):
        os.makedirs(res_pth)

    netE.load_state_dict(torch.load('./netE.pth'))
    netE.eval()
    netG.load_state_dict(torch.load('./netG.pth'))
    netG.eval()

    for epoch in range(3):
        for i, (src_img, src_age) in enumerate(dataloader):
            print(i)
            
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

            netR.zero_grad()
            pred_age = netR(loss_values).view(-1)
            loss_r = MSE(pred_age.float(), src_age.float())
            loss_r.backward()
            optimR.step()
            
            if not i % 100:
                with open(res_pth + 'train_result.csv', 'a', encoding='utf-8', newline='') as F:
                    writer = csv.writer(F)
                    writer.writerow([i, tl(loss_values[0]), tl(src_age[0]), tl(pred_age[0]), tl(loss_r)])

    torch.save(netR.state_dict(), res_pth + 'netR.pth')


if __name__ == '__main__':
    main()
