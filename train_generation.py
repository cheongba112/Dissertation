import os, random, csv, time, re, shutil
import numpy as np
# from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from torchvision import utils
from torchvision.models import googlenet

from options import *
from misc import *
from get_dataset import *
from models import *


val_bth = 8
res_pth = './generation_result/'

# dataset
dataset = get_dataset(opt.dataroot)
validset = get_dataset(opt.dataroot + '_valid')

# dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=1)
validloader = torch.utils.data.DataLoader(validset, batch_size=val_bth, shuffle=False, num_workers=1)

# use cpu or gpu as device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model
netE    = Encoder().to(device)
netG    = Generator().to(device)
netvecD = VectorDiscriminator().to(device)
googlenet = googlenet(pretrained=True).to(device)
netageD = AgeDiscriminator().to(device)
netageC = AgeClassifier().to(device)

# model weights initialise
netE.apply(weights_init)
netG.apply(weights_init)
netvecD.apply(weights_init)
netageD.apply(weights_init)
netageC.apply(weights_init)

# loss function
L1  = nn.L1Loss().to(device)
MSE = nn.MSELoss().to(device)
CE  = nn.CrossEntropyLoss().to(device)
BCE = nn.BCELoss().to(device)
KL  = nn.KLDivLoss().to(device)

# optimizer
optimE    = optim.Adam(netE.parameters(),    lr=0.0002, betas=(0.5, 0.999))
optimG    = optim.Adam(netG.parameters(),    lr=0.0002, betas=(0.5, 0.999))
optimvecD = optim.Adam(netvecD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimageD = optim.Adam(netageD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimageC = optim.Adam(netageC.parameters(), lr=0.0002, betas=(0.5, 0.999))

# training
if __name__ == '__main__':
    # start = time.time()
    
    if not os.path.exists(res_pth):
        os.makedirs(res_pth + 'pics')

    with open(res_pth + 'train_result.csv', 'a', encoding='utf-8', newline='') as F:
        w = csv.writer(F)
        w.writerow(['epoch', 'i', 'vecT', 'vecF', 'ageDT', 'ageDF', 'C', 'l1', 'tv', 'vec', 'id', 'aged', 'agec', 'aim'])

    for epoch in range(opt.epoch_num):
        print('Epoch: %d' % (epoch))
        
        for i, (src_img, src_age) in enumerate(dataloader):
            print('Batch: %d' % (i))
            
            # prepare batch data
            src_img = src_img.to(device)

            for k in range(src_age.size()[0]):
                if   src_age[k] <= 5:  src_age[k] = 0
                elif src_age[k] <= 10: src_age[k] = 1
                elif src_age[k] <= 15: src_age[k] = 2
                elif src_age[k] <= 20: src_age[k] = 3
                elif src_age[k] <= 30: src_age[k] = 4
                elif src_age[k] <= 40: src_age[k] = 5
                elif src_age[k] <= 50: src_age[k] = 6
                elif src_age[k] <= 60: src_age[k] = 7
                elif src_age[k] <= 70: src_age[k] = 8
                else:                  src_age[k] = 9

            src_age = src_age.to(device)
            src_age_onehot = torch.zeros(src_age.size()[0], 10).to(device)
            for j, age in enumerate(src_age):
                src_age_onehot[j][age] = 1.
            
            # syn_age = torch.LongTensor(src_age.size()).fill_(np.random.randint(10)).to(device)
            # syn_age_onehot = torch.zeros(src_age.size()[0], 10).to(device)
            # for j, age in enumerate(syn_age):
            #     syn_age_onehot[j][age] = 1.

            pri_vec = torch.FloatTensor(src_age.size()[0], 50).uniform_(0, 1).to(device)

            real_label  = torch.full(src_age.size(), 1, device=device, dtype=torch.float)
            false_label = torch.full(src_age.size(), 0, device=device, dtype=torch.float)
            
            infocol = [epoch, i]
            
            # generate synthesized images
            syn_vec = netE(src_img, src_age_onehot)
            syn_img = netG(syn_vec, src_age_onehot)

            # train vector D
            netvecD.zero_grad()
            output_vecD_T = netvecD(pri_vec).view(-1)
            output_vecD_F = netvecD(syn_vec.detach()).view(-1)
            loss_vecD_T = BCE(output_vecD_T, real_label)
            loss_vecD_F = BCE(output_vecD_F, false_label)
            loss_vecD = loss_vecD_T + loss_vecD_F
            loss_vecD.backward()
            optimvecD.step()
            infocol.append(tl(loss_vecD_T))
            infocol.append(tl(loss_vecD_F))

            # train age D
            netageD.zero_grad()
            output_ageD_T = netageD(src_img, src_age_onehot).view(-1)
            output_ageD_F = netageD(syn_img.detach(), src_age_onehot).view(-1)
            loss_ageD_T = BCE(output_ageD_T, real_label)
            loss_ageD_F = BCE(output_ageD_F, false_label)
            loss_ageD = loss_ageD_T + loss_ageD_F
            loss_ageD.backward()
            optimageD.step()
            infocol.append(tl(loss_ageD_T))
            infocol.append(tl(loss_ageD_F))

            # train age C
            netageC.zero_grad()
            output_ageC = netageC(src_img)
            loss_ageC = CE(output_ageC, src_age)
            loss_ageC.backward()
            optimageC.step()
            infocol.append(tl(loss_ageC))


            # train encoder and generator
            netE.zero_grad()
            netG.zero_grad()
            
            # L1 loss
            loss_l1_G = L1(syn_img, src_img)
            infocol.append(tl(loss_l1_G))
            
            # tv loss
            loss_tv_G = TV_Loss(syn_img)
            infocol.append(tl(loss_tv_G))

            # vector D loss
            output_vec_G = netvecD(syn_vec).view(-1)
            loss_vec_G = BCE(output_vec_G, real_label)
            infocol.append(tl(loss_vec_G))

            # ID loss
            src_id_vec = googlenet(src_img)
            syn_id_vec = googlenet(syn_img)
            loss_id_G = L1(syn_id_vec, src_id_vec)
            infocol.append(tl(loss_id_G))

            # age D loss
            output_aged_G = netageD(syn_img, src_age_onehot).view(-1)
            loss_aged_G = BCE(output_aged_G, real_label)
            infocol.append(tl(loss_aged_G))

            # age C loss
            output_agec_G = netageC(syn_img)
            loss_agec_G = CE(output_agec_G, src_age)
            infocol.append(tl(loss_agec_G))

            loss_G = loss_l1_G + loss_tv_G + 0.01 * loss_vec_G + 0.001 * loss_id_G + 0.001 * loss_aged_G + 0.001 * loss_agec_G
            loss_G.backward()
            optimE.step()
            optimG.step()
            infocol.append(tl(loss_G + loss_vecD + loss_ageD + loss_ageC))
            
            # save loss info every 500 batchs
            if not i % 500:
                with open(res_pth + 'train_result.csv', 'a', encoding='utf-8', newline='') as F:
                    writer = csv.writer(F)
                    writer.writerow(infocol)

        # generate test result every epoch
        for src_img, _ in validloader:
            src_img = src_img.to(device)
            imgs = src_img
            for i in [2, 4, 5, 6, 7, 8]:
                syn_age_onehot = torch.zeros(val_bth, 10).to(device)
                for j in range(val_bth):
                    syn_age_onehot[j][i] = 1.
                syn_vec = netE(src_img, syn_age_onehot)
                syn_img = netG(syn_vec, syn_age_onehot)
                imgs = torch.cat((imgs, syn_img), 0)
            utils.save_image(imgs, './generation_result/pics/C_%d.jpg' % (epoch), normalize=True)

    # print loss line charts
    # with open(res_pth + 'train_result.csv', 'r') as F:
    #     r = csv.reader(F)
    #     data = [row for row in r]
    #     for i in range(2, len(data[0])):  # skip epoch and batch columns
    #         col = [row[i] for row in data]
    #         label = col[0]
    #         value = []
    #         for d in col[1:]:
    #             value.append(float(d))
    #         plt.plot(range(len(value)), value, '-')
    #         plt.ylabel(label)
    #         plt.savefig(res_pth + label + '.jpg')
    #         plt.clf()
    
    torch.save(netE.state_dict(), './netE.pth')
    torch.save(netG.state_dict(), './netG.pth')
