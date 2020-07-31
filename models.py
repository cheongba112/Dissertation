import os, random
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.models import vgg19

from get_dataset import CACD_Dataset
from options import opt

# dataset
dataset = CACD_Dataset('./14')

# dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=1)

# use cpu or gpu as device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Models
class Gen(nn.Module):
    def __init__(self):
        super(Gen, self).__init__()
        self.embed = nn.Embedding(80, 49)  # from 0 to 79 years old, 7 * 7 vector
        self.encode = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [1, 64, 7, 7]
        )
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(65, 64, 3, stride=2),  # 15
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 3, stride=2),  # 31
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 2, stride=2),  # 62
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 3, stride=2),  # 125
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 2, stride=2),   # 250
            nn.ReLU(True),
        )

    def forward(self, x, con, batch_size):
        latent_vec = self.encode(x)
        latent_vec = torch.cat((torch.reshape(self.embed(con), (batch_size, 1, 7, 7)), latent_vec), 1)
        output = self.decode(latent_vec)
        # output = output.view(3, 250, 250)  # not working for mini-batch
        return output

class Age_Dis(nn.Module):
    def __init__(self):
        super(Age_Dis, self).__init__()
        self.embed = nn.Embedding(80, 49)
        self.encode = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.regre = nn.Sequential(
            nn.Conv2d(65, 64, 3),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, con, batch_size):
        latent_vec = self.encode(x)
        latent_vec = torch.cat((torch.reshape(self.embed(con), (batch_size, 1, 7, 7)), latent_vec), 1)
        output = self.regre(latent_vec)
        return output

class Id_Dis(nn.Module):
    '''
    Input: 6 * 250 * 250 (img1, img2)
    '''
    def __init__(self):
        super(Id_Dis, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, 3),    # 248
            nn.ReLU(True),
            nn.MaxPool2d(2),        # 124
            nn.Conv2d(64, 64, 3),   # 122
            nn.ReLU(True),
            nn.MaxPool2d(2),        # 61
            nn.Conv2d(64, 64, 3),   # 59
            nn.ReLU(True),
            nn.MaxPool2d(2),        # 29
            nn.Conv2d(64, 64, 3),   # 27
            nn.ReLU(True),
            nn.MaxPool2d(3),        # 9
            nn.Flatten(),
            nn.Linear(64*9*9, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, src_img, tgt_img):
        input_img = torch.cat((src_img, tgt_img), 1)
        output = self.model(input_img)
        return output

# class Age_Reg(nn.Module):



gen     = Gen().to(device)
age_dis = Age_Dis().to(device)
id_dis  = Id_Dis().to(device)

loss_func = nn.BCELoss()

optim_gen     = optim.Adam(gen.parameters(),     lr=0.0002, betas=(0.5, 0.999))
optim_age_dis = optim.Adam(age_dis.parameters(), lr=0.0002, betas=(0.5, 0.999))
optim_id_dis  = optim.Adam(id_dis.parameters(),  lr=0.0002, betas=(0.5, 0.999))

# -----------------training----------------
# 200729

if __name__ == '__main__':
    for epoch in range(10):
        for (src_img, src_age, tgt_img) in dataloader:  # 100, Tensor
            # put mini-batch into device
            src_img = src_img.to(device)
            src_age = src_age.to(device)
            tgt_img = tgt_img.to(device)
            
            # get batch length
            batch_len = src_age.size()[0]

            # ------ train age D
            age_dis.zero_grad()
            
            # train true images
            label = torch.full(src_age.size(), 1, device=device)
            output = age_dis(src_img, src_age, batch_len).view(-1)
            loss = loss_func(output, label)
            loss.backward()

            # train false images
            syn_age = torch.tensor(np.random.randint(1, 80, (src_age.size())), dtype=torch.int64).to(device)
            syn_img = gen(src_img, syn_age, batch_len)
            
            label.fill_(0)
            output = age_dis(syn_img.detach(), syn_age, batch_len).view(-1)  # use detach() to fix G
            loss = loss_func(output, label)
            loss.backward()

            # update weights
            optim_age_dis.step()

            # ------ train G
            gen.zero_grad()
            label.fill_(1)

            output = age_dis(syn_img, syn_age, batch_len).view(-1)
            loss = loss_func(output, label)
            loss.backward()

            optim_gen.step()

            print(loss)


'''
# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()        # Does the update
'''














# 200727
# tooooo slooooooow
# for img in dataset:
#     lst = [x for x in dataset if x[2] == img[2]]
#     p(lst[random.randint(0, len(lst) - 1)][2])

# ----------------test---------------------
# 200725

# print(dataset[0][2])

# a = [x for x in dataset if x[2] == 'Steven_Anthony_Lawrence']
# print(a)

# for i, (img, age, name) in enumerate(a):
#     print(i, age, name)

# print(dataset.__len__())


# img1 = Image.open('test1.jpg')
# img2 = Image.open('test2.jpg')

# img2tensor = transforms.ToTensor()

# img1 = img2tensor(img1).resize_(1, 3, 250, 250)
# img2 = img2tensor(img2).resize_(1, 3, 250, 250)

# id_dis = Id_Dis()
# age_dis = Age_Dis()

# # out = id_dis(img1, img2)
# out = age_dis(img1, torch.tensor([14]))

# print(out.size())

# --------------Rough Test-------------------
# 200723

# img = Image.open('test.jpg')
# img2tensor = transforms.ToTensor()
# img = img2tensor(img).resize_(1, 3, 250, 250)

# # print(img.size())
# # [3, 250, 250]

# gen = Gen()
# out = gen(img, torch.tensor([1]))
# print(out)

# # out = out.resize_(3, 250, 250)

# unloader = transforms.ToPILImage()
# out = unloader(out)
# out.save('res.jpg')
