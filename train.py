import os, random
import numpy as np
from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import utils

from models import *
from get_dataset import get_dataset
from options import opt

# dataset
dataset = get_dataset(opt.dataroot)

# dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=1)

# use cpu or gpu as device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model
gen     = Gen().to(device)
age_dis = AgeD().to(device)
id_dis  = IdD().to(device)

# loss function
loss_func = nn.BCELoss()

# optimizer
optim_gen     = optim.Adam(gen.parameters(),     lr=0.0002, betas=(0.5, 0.999))
optim_age_dis = optim.Adam(age_dis.parameters(), lr=0.0002, betas=(0.5, 0.999))
optim_id_dis  = optim.Adam(id_dis.parameters(),  lr=0.0002, betas=(0.5, 0.999))


# -----------------training----------------
if __name__ == '__main__':
    for epoch in range(opt.epoch_num):
        for i, (src_img, src_age, tgt_img) in enumerate(dataloader):
            # put batch into device
            src_img = src_img.to(device)
            tgt_img = tgt_img.to(device)
            src_age = src_age.to(device)
            syn_age = torch.tensor(np.random.randint(10, 71, src_age.size()),
                                   dtype=torch.int64).to(device)
            label = torch.full(src_age.size(), 1, device=device)

            # get batch length
            batch_len = src_age.size()[0]

            # generate synthesized images
            syn_img = gen(src_img, syn_age)

            # ------------------------------------------------------------------
            # train age D
            # train true images
            label.fill_(1)
            output_age_T = age_dis(src_img, src_age).view(-1)
            loss_age_T = loss_func(output_age_T, label)
            
            # train false images
            label.fill_(0)
            # use detach() to fix G
            output_age_F = age_dis(syn_img.detach(), syn_age).view(-1)
            loss_age_F = loss_func(output_age_F, label)
            
            # update weights
            age_dis.zero_grad()
            loss_age = (loss_age_T + loss_age_F) / 2
            loss_age.backward()
            optim_age_dis.step()

            # ------------------------------------------------------------------
            # train id D
            # train true images
            label.fill_(1)
            output_id_T = id_dis(src_img, tgt_img).view(-1)
            loss_id_T = loss_func(output_id_T, label)
            
            # train false images
            label.fill_(0)
            output_id_F = id_dis(src_img, syn_img.detach()).view(-1)
            loss_id_F = loss_func(output_id_F, label)
            
            # update weights
            id_dis.zero_grad()
            loss_id = (loss_id_T + loss_id_F) / 2
            loss_id.backward()
            optim_id_dis.step()

            # ------------------------------------------------------------------
            # train G
            label.fill_(1)
            output_age_g = age_dis(syn_img, syn_age).view(-1)
            loss_age_g = loss_func(output_age_g, label)

            output_id_g = id_dis(syn_img, syn_img).view(-1)
            loss_id_g = loss_func(output_id_g, label)
            
            gen.zero_grad()
            loss_g = (loss_age_g + loss_id_g) / 2
            loss_g.backward()
            optim_gen.step()
            
            if batch_len < opt.batch_size:  # last batch of each epoch
                utils.save_image(syn_img, './pics/%d_%d.jpg' % (epoch, i), normalize=True)
                print('epoch: %d\nbatch: %d\nage_loss: %f\nid_loss: %f\ng_loss: %f\n'
                      % (epoch, i, loss_age.data[0], loss_id.data[0], loss_g.data[0]))

        # break

    # torch.save(gen.state_dict(), 'g.pth')

'''
line72
RuntimeError: one of the variables needed for gradient computation has been 
modified by an inplace operation: [torch.cuda.FloatTensor [32]] is at version 
3; expected version 2 instead. Hint: enable anomaly detection to find the 
operation that failed to compute its gradient, with 
torch.autograd.set_detect_anomaly(True).
'''


# loss_age.backward(retain_graph=True)
# RuntimeError: Trying to backward through the graph a second time, but the
# buffers have already been freed. Specify retain_graph=True when calling
# backward the first time.

'''
torch.save(model.state_dict(), filepath)

#Later to restore:
model.load_state_dict(torch.load(filepath))
model.eval()
'''

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
