import os, random, csv, time
import numpy as np
from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import utils

from models import *
from get_dataset import get_dataset
from options import opt


dataset = get_dataset(opt.dataroot)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

netE = Encoder().to(device)
netG = Generator().to(device)
netR = FinalAgeRegressor().to(device)


def main():
    # load pre-trained weights
    if opt.netE_path != None and opt.netG_path != None:
        netE.load_state_dict(torch.load(opt.netE_path))
        netE.eval()
        netG.load_state_dict(torch.load(opt.netG_path))
        netG.eval()
    else:
        print('No pre-trained models')
        return

    for epoch in range(opt.epoch_num):
        for i, (src_img, src_age, _) in enumerate(dataloader):
            print(epoch, i)
            



if __name__ == '__main__':
    main()