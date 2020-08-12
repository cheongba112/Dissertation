'''
ageR -> ageD
cancel ref_img
same syn_age in each batch
'''
import os, random, csv, time
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import utils

from models import *
from get_dataset import get_dataset
from options import opt

# tensor to list
def tl(tensor):
    return str(tensor.tolist())

# weights initialise function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)

# dataset
opt.dataroot = './cacd_lite'  # local debug option
dataset = get_dataset(opt.dataroot)

# dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=1)

# use cpu or gpu as device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model
netE    = Encoder().to(device)
netG    = Generator().to(device)
netageD = AgeDiscriminator().to(device)
netimgD = ImageDiscriminator().to(device)
netvecD = VectorDiscriminator().to(device)

# model weights initialise
netE.apply(weights_init)
netG.apply(weights_init)
netageD.apply(weights_init)
netimgD.apply(weights_init)
netvecD.apply(weights_init)

# loss function
BCE = nn.BCELoss().to(device)
L1  = nn.L1Loss().to(device)
MSE = nn.MSELoss().to(device)

# optimizer
optimE    = optim.Adam(netE.parameters(),    lr=0.0002, betas=(0.5, 0.999))
optimG    = optim.Adam(netG.parameters(),    lr=0.0002, betas=(0.5, 0.999))
optimageD = optim.Adam(netageD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimimgD = optim.Adam(netimgD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimvecD = optim.Adam(netvecD.parameters(), lr=0.0002, betas=(0.5, 0.999))

# training
if __name__ == '__main__':
    # start = time.time()
    # with open('./train_result/train_result.csv', 'a', encoding='utf-8', newline='') as F:
    #     w = csv.writer(F)
    #     w.writerow(['epoch', 'batch', 'loss_ageD', 'loss_vecD', 'loss_imgD', 'loss_G', 'syn_age'])

    for epoch in range(opt.epoch_num):
        print('Epoch: %d' % (epoch))
        
        for i, (src_img, src_age, ref_img) in enumerate(dataloader):
            print('Batch: %d' % (i))

            # put batch into device
            src_img = src_img.to(device)
            # ref_img = ref_img.to(device)
            src_age = src_age.to(device)
            syn_age = torch.LongTensor(src_age.size()).fill_(np.random.randint(10, 71)).to(device)
            pri_vec = torch.FloatTensor(src_age.size()[0], 100).uniform_(-1, 1).to(device)
            real_label  = torch.full(src_age.size(), 1, device=device)  # warning
            false_label = torch.full(src_age.size(), 0, device=device)  # warning

            # generate synthesized images
            syn_vec = netE(src_img)
            syn_img = netG(syn_vec, syn_age)

            # ------------------------------------------------------------------
            # train age D
            netageD.zero_grad()
            output_age_T = netageD(src_img, src_age).view(-1)
            output_age_F = netageD(syn_img.detach(), syn_age).view(-1)
            loss_ageD = BCE(output_age_T, real_label) + BCE(output_age_F, false_label)
            loss_ageD.backward()
            optimageD.step()

            # ------------------------------------------------------------------
            # train image D
            netimgD.zero_grad()
            output_img_T = netimgD(src_img).view(-1)
            output_img_F = netimgD(syn_img.detach()).view(-1)
            loss_imgD = BCE(output_img_T, real_label) + BCE(output_img_F, false_label)
            loss_imgD.backward()
            optimimgD.step()

            # ------------------------------------------------------------------
            # train vector D
            netvecD.zero_grad()
            output_vec_T = netvecD(pri_vec).view(-1)
            output_vec_F = netvecD(syn_vec.detach()).view(-1)
            loss_vecD = BCE(output_vec_T, real_label) + BCE(output_vec_F, false_label)
            loss_vecD.backward()
            optimvecD.step()

            # ------------------------------------------------------------------
            # train encoder and generator
            netE.zero_grad()
            netG.zero_grad()
            
            # L1 loss
            loss_l1_G = L1(syn_img, src_img)

            # age D loss
            output_age_G = netageD(syn_img, syn_age).view(-1)
            loss_age_G = BCE(output_age_G, real_label)

            # image D loss
            output_img_G = netimgD(syn_img).view(-1)
            loss_img_G = BCE(output_img_G, real_label)

            # vector D loss
            output_vec_G = netvecD(syn_vec).view(-1)
            loss_vec_G = BCE(output_vec_G, real_label)

            loss_G = loss_l1_G + 0.0001 * loss_age_G + loss_img_G + 0.01 * loss_vec_G
            loss_G.backward()

            optimE.step()
            optimG.step()

            # print(time.time() - start)
            # print(epoch, i, tl(loss_ageD), tl(loss_vecD), tl(loss_imgD), tl(loss_G), tl(syn_age[0]))

            # ------------------------------------------------------------------
            if not i % 100:
                utils.save_image(syn_img, './train_result/pics/%d_%d.jpg' % (epoch, i), normalize=True)
                with open('./train_result/train_result.csv', 'a', encoding='utf-8', newline='') as F:
                    writer = csv.writer(F)
                    writer.writerow([epoch, i, 
                        tl(loss_ageD), 
                        tl(loss_vecD), 
                        tl(loss_imgD), 
                        tl(loss_G), 
                        tl(syn_age[0])])

            break
        break

    torch.save(netE.state_dict(),    './train_result/netE.pth')
    torch.save(netG.state_dict(),    './train_result/netG.pth')
    torch.save(netageD.state_dict(), './train_result/netageD.pth')
    torch.save(netimgD.state_dict(), './train_result/netimgD.pth')
    torch.save(netvecD.state_dict(), './train_result/netvecD.pth')
