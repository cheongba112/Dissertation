import os, random
import numpy as np
from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from models import *
from get_dataset import CACD_Dataset
from options import opt

# dataset
dataset = CACD_Dataset('./14')

# dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=1)

# use cpu or gpu as device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model
gen     = Gen().to(device)
age_dis = Age_Dis().to(device)
id_dis  = Id_Dis().to(device)

# loss function
loss_func = nn.BCELoss()

# optimizer
optim_gen     = optim.Adam(gen.parameters(),     lr=0.0002, betas=(0.5, 0.999))
optim_age_dis = optim.Adam(age_dis.parameters(), lr=0.0002, betas=(0.5, 0.999))
optim_id_dis  = optim.Adam(id_dis.parameters(),  lr=0.0002, betas=(0.5, 0.999))

# -----------------training----------------
# 200731

if __name__ == '__main__':
    for epoch in range(10):
        for (src_img, src_age, tgt_img) in dataloader:  # 100, Tensor
            # put mini-batch into device
            src_img = src_img.to(device)
            src_age = src_age.to(device)
            tgt_img = tgt_img.to(device)
            
            # get batch length
            batch_len = src_age.size()[0]

            # train age D
            age_dis.zero_grad()
            # train true images
            label = torch.full(src_age.size(), 1, device=device)  # label initialized with 1
            output = age_dis(src_img, src_age, batch_len).view(-1)
            loss = loss_func(output, label)
            loss.backward()
            # generate images
            syn_age = torch.tensor(np.random.randint(1, 80, (src_age.size())), dtype=torch.int64).to(device)
            syn_img = gen(src_img, syn_age, batch_len)
            # train false images
            label.fill_(0)
            output = age_dis(syn_img.detach(), syn_age, batch_len).view(-1)  # use detach() to fix G
            loss = loss_func(output, label)
            loss.backward()
            # update weights
            optim_age_dis.step()

            # train id D
            id_dis.zero_grad()
            # train true images
            label.fill_(1)
            output = id_dis(src_img, tgt_img).view(-1)
            loss = loss_func(output, label)
            loss.backward()
            # train false images
            label.fill_(0)
            output = id_dis(src_img, syn_img.detach()).view(-1)
            loss = loss_func(output, label)
            loss.backward()
            # update weights
            optim_id_dis.step()

            # train G
            gen.zero_grad()
            label.fill_(1)
            output = age_dis(syn_img, syn_age, batch_len).view(-1)
            loss_age = loss_func(output, label)
            # loss.backward(retain_graph=True)
            output = id_dis(syn_img, syn_img).view(-1)
            loss_id = loss_func(output, label)
            loss = loss_age + loss_id
            loss.backward()
            optim_gen.step()

            # print(loss)
        # break

# RuntimeError: Trying to backward through the graph a second time, but the
# buffers have already been freed. Specify retain_graph=True when calling
# backward the first time.

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
