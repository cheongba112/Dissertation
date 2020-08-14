import os, random, csv, time
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import utils

from models import *
from get_dataset import get_dataset
from options import opt
from misc import *

local_debug = True

# dataset
if local_debug:
    opt.dataroot = './cacd_lite'
dataset = get_dataset(opt.dataroot)

# dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=1)

# use cpu or gpu as device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model
netE    = Encoder().to(device)
netG    = Generator().to(device)
netageR = AgeRegressor().to(device)
netimgD = ImageDiscriminator().to(device)
netvecD = VectorDiscriminator().to(device)

# model weights initialise
netE.apply(weights_init)
netG.apply(weights_init)
netageR.apply(weights_init)
netimgD.apply(weights_init)
netvecD.apply(weights_init)

# loss function
BCE = nn.BCELoss().to(device)
L1  = nn.L1Loss().to(device)
MSE = nn.MSELoss().to(device)

# optimizer
optimE    = optim.Adam(netE.parameters(),    lr=0.0002, betas=(0.5, 0.999))
optimG    = optim.Adam(netG.parameters(),    lr=0.0002, betas=(0.5, 0.999))
optimageR = optim.Adam(netageR.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimimgD = optim.Adam(netimgD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimvecD = optim.Adam(netvecD.parameters(), lr=0.0002, betas=(0.5, 0.999))

# training
if __name__ == '__main__':
    # start = time.time()

    if not local_debug:
        with open('./train_result/train_result.csv', 'a', encoding='utf-8', newline='') as F:
            w = csv.writer(F)
            w.writerow(['epoch', 'batch', 
                'loss_ageR', 
                'loss_imgD_T', 'loss_imgD_F', 
                'loss_vecD_T', 'loss_vecD_F', 
                'loss_l1_G', 'loss_age_G', 'loss_img_G', 'loss_vec_G', 'loss_tv_G', 
                'syn_age'])

    for epoch in range(opt.epoch_num):
        print('Epoch: %d' % (epoch))
        
        for i, (src_img, src_age, _) in enumerate(dataloader):
            print('Batch: %d' % (i))
            
            # prepare batch data
            src_age = src_age.to(device)
            src_img = src_img.to(device)
            
            pri_vec = torch.FloatTensor(src_age.size()[0], 50).uniform_(-1, 1).to(device)
            
            real_label  = torch.full(src_age.size(), 1, device=device)  # warning
            false_label = torch.full(src_age.size(), 0, device=device)  # warning
            
            src_age_onehot = - torch.ones(src_age.size()[0], 100).to(device)
            for j, age in enumerate(src_age):
                src_age_onehot[j][age - 1] = 1.  # 0th pos -> 1 year old

            syn_age = torch.LongTensor(src_age.size()).fill_(np.random.randint(10, 71)).to(device)  # 10 - 70 years old
            syn_age_onehot = - torch.ones(src_age.size()[0], 100).to(device)
            for j, age in enumerate(syn_age):
                syn_age_onehot[j][age - 1] = 1.

            # generate synthesized images
            syn_vec = netE(src_img)
            syn_img = netG(syn_vec, syn_age_onehot)

            # ------------------------------------------------------------------
            # train age Regressor
            netageR.zero_grad()
            output_age = netageR(src_img).view(-1)
            loss_ageR = MSE(output_age.float(), src_age.float())
            loss_ageR.backward()
            optimageR.step()

            # ------------------------------------------------------------------
            # train image D
            netimgD.zero_grad()
            output_img_T = netimgD(src_img).view(-1)
            output_img_F = netimgD(syn_img.detach()).view(-1)
            loss_imgD_T = BCE(output_img_T, real_label)
            loss_imgD_F = BCE(output_img_F, false_label)
            loss_imgD = loss_imgD_T + loss_imgD_F
            loss_imgD.backward()
            optimimgD.step()

            # ------------------------------------------------------------------
            # train vector D
            netvecD.zero_grad()
            output_vec_T = netvecD(pri_vec).view(-1)
            output_vec_F = netvecD(syn_vec.detach()).view(-1)
            loss_vecD_T = BCE(output_vec_T, real_label)
            loss_vecD_F = BCE(output_vec_F, false_label)
            loss_vecD = loss_vecD_T + loss_vecD_F
            loss_vecD.backward()
            optimvecD.step()

            # ------------------------------------------------------------------
            # train encoder and generator
            netE.zero_grad()
            netG.zero_grad()
            
            # L1 loss
            loss_l1_G = L1(syn_img, src_img)

            # age R loss
            output_age_G = netageR(syn_img).view(-1)
            loss_age_G = MSE(output_age_G.float(), syn_age.float())
            loss_age_G = loss_age_G * 0.01

            # image D loss
            output_img_G = netimgD(syn_img).view(-1)
            loss_img_G = BCE(output_img_G, real_label)

            # vector D loss
            output_vec_G = netvecD(syn_vec).view(-1)
            loss_vec_G = BCE(output_vec_G, real_label)

            # tv loss
            loss_tv_G = TV_Loss(syn_img)

            loss_G = loss_l1_G + 0.0001 * loss_age_G + 0.0001 * loss_img_G + 0.01 * loss_vec_G + loss_tv_G
            loss_G.backward()

            optimE.step()
            optimG.step()

            # print(time.time() - start)

            # ------------------------------------------------------------------
            if local_debug:
                print(epoch, i, 
                    tl(loss_ageR), 
                    tl(loss_imgD_T), tl(loss_imgD_F), 
                    tl(loss_vecD_T), tl(loss_vecD_F), 
                    tl(loss_l1_G), tl(loss_age_G), tl(loss_img_G), tl(loss_vec_G), tl(loss_tv_G), 
                    tl(syn_age[0]))
                break
            else:
                if not i % 100:
                    utils.save_image(syn_img, './train_result/pics/%d_%d.jpg' % (epoch, i), normalize=True)
                    with open('./train_result/train_result.csv', 'a', encoding='utf-8', newline='') as F:
                        writer = csv.writer(F)
                        writer.writerow([epoch, i, 
                            tl(loss_ageR), 
                            tl(loss_imgD_T), tl(loss_imgD_F), 
                            tl(loss_vecD_T), tl(loss_vecD_F), 
                            tl(loss_l1_G), tl(loss_age_G), tl(loss_img_G), tl(loss_vec_G), tl(loss_tv_G), 
                            tl(syn_age[0])])

        if local_debug:
            break

    if not local_debug:
        torch.save(netE.state_dict(),    './train_result/netE.pth')
        torch.save(netG.state_dict(),    './train_result/netG.pth')
        torch.save(netageR.state_dict(), './train_result/netageR.pth')
        torch.save(netimgD.state_dict(), './train_result/netimgD.pth')
        torch.save(netvecD.state_dict(), './train_result/netvecD.pth')
