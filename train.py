import os, random, csv, time
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import utils

from models.models import *
from get_dataset import get_dataset
from options import opt

# tensor to list
def tl(tensor):
    return str(tensor.tolist())

# dataset
dataset = get_dataset(opt.dataroot)

# dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=1)

# use cpu or gpu as device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model
netE    = Encoder().to(device)
netG    = Decoder().to(device)
netageR = AgeRegressor().to(device)
netimgD = ImageDiscriminator().to(device)
netvecD = VectorDiscriminator().to(device)

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
    # with open('./train_result/train_result.csv', 'a', encoding='utf-8', newline='') as F:
    #     w = csv.writer(F)
    #     w.writerow(['epoch', 'batch', 'loss_ageR', 'loss_vecD', 'loss_imgD', 'loss_G', 'syn_age'])

    for epoch in range(opt.epoch_num):
        print('Epoch: %d' % (epoch))
        
        for i, (src_img, src_age, ref_img) in enumerate(dataloader):
            print('Batch: %d' % (i))

            # get batch length
            batch_len = src_age.size()[0]

            # put batch into device
            src_img = src_img.to(device)
            ref_img = ref_img.to(device)
            src_age = src_age.to(device)
            syn_age = torch.tensor(np.random.randint(10, 71, src_age.size()),
                                   dtype=torch.int64).to(device)
            pri_vec = torch.FloatTensor((opt.batch_len, 100))
                           .uniform_(-1, 1).to(device)
            real_label  = torch.full(src_age.size(), 1, device=device)
            false_label = torch.full(src_age.size(), 0, device=device)

            # generate synthesized images
            syn_vec = netE(src_img)
            syn_img = netG(syn_vec, syn_age)

            # ------------------------------------------------------------------
            # train age regressor
            netageR.zero_grad()
            output_ageR = netageR(syn_img.detach())
            loss_ageR = MSE(output_ageR, src_age)
            loss_ageR.backward()
            optimageR.step()

            # ------------------------------------------------------------------
            # train vector D
            netvecD.zero_grad()
            output_vec_T = netvecD(pri_vec)
            output_vec_F = netvecD(syn_vec.detach())
            loss_vecD = BCE(output_vec_T, real_label) + BCE(output_vec_F, false_label)
            loss_vecD.backward()
            optimvecD.step()

            # ------------------------------------------------------------------
            # train image D
            netimgD.zero_grad()
            output_img_T = netimgD(src_img)  # netimgD(ref_img)
            output_img_F = netimgD(syn_img.detach())
            loss_imgD = BCE(output_img_T, real_label) + BCE(output_img_F, false_label)
            loss_imgD.backward()
            optimimgD.step()

            # ------------------------------------------------------------------
            # train encoder and generator
            netE.zero_grad()
            netG.zero_grad()
            
            # L1 loss
            loss_l1_G = L1(syn_img, ref_img)  # L1(syn_img, src_img)

            # age regressor loss
            output_ageR_G = netageR(syn_img)
            loss_ageR_G = MSE(output_ageR_G, syn_age)

            # image D loss
            output_imgD_G = netimgD(syn_img)
            loss_imgD_G = BCE(output_imgD_G, real_label)

            # vector D loss
            output_vecD_G = netvecD(syn_vec)
            loss_vecD_G = BCE(output_vecD_G, real_label)

            loss_G = loss_imgD_G + 0.01 * loss_l1_G + 0.01 * loss_vecD_G + 0.01 * loss_ageR
            loss_G.backward()
            optimE.step()
            optimG.step()

            # print(time.time() - start)
            print(epoch, i, tl(loss_ageR), tl(loss_vecD), tl(loss_imgD), tl(loss_G), tl(syn_age))

            # ------------------------------------------------------------------
            # if not i % 100:
            #     utils.save_image(syn_img, './train_result/pics/%d_%d.jpg' % (epoch, i), normalize=True)
            #     with open('./train_result/train_result.csv', 'a', encoding='utf-8', newline='') as F:
            #         writer = csv.writer(F)
            #         writer.writerow([epoch, i, 
            #             tl(loss_ageR), 
            #             tl(loss_vecD), 
            #             tl(loss_imgD), 
            #             tl(loss_G), 
            #             tl(syn_age)])

            break
        break

    # torch.save(netE.state_dict(),    './train_result/netE.pth')
    # torch.save(netG.state_dict(),    './train_result/netG.pth')
    # torch.save(netageR.state_dict(), './train_result/netageR.pth')
    # torch.save(netimgD.state_dict(), './train_result/netimgD.pth')
    # torch.save(netvecD.state_dict(), './train_result/netvecD.pth')
