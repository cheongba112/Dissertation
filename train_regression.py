import os, csv, argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from misc import *
from get_dataset import *
from models import *

# adjustable options
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot',    required=False, type=str, default='./UTKFace',  help='path to dataset')
parser.add_argument('--regre_epoch', required=False, type=int, default=3,            help='regression training epoch number')
parser.add_argument('--netE_path',   required=False, type=str, default='./netE.pth', help='path to pre-trained encoder model')
parser.add_argument('--netG_path',   required=False, type=str, default='./netG.pth', help='path to pre-trained generator model')
parser.add_argument('--save_Rpth',   required=False, type=str, default='./netR.pth', help='regressor save path')
opt = parser.parse_args()

# fised options
res_pth = './regression_result/'

dataset = get_dataset(opt.dataroot)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)  # pure SGD

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

    netE.load_state_dict(torch.load(opt.netE_path))
    netE.eval()
    netG.load_state_dict(torch.load(opt.netG_path))
    netG.eval()

    for epoch in range(opt.regre_epoch):
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

    torch.save(netR.state_dict(), opt.save_Rpth)


if __name__ == '__main__':
    main()
