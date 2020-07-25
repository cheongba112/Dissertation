import os, numpy, argparse
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.models import vgg19

from get_dataset import *

# Create the dataset
dataset = CACD_Dataset('./14')

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=100,
                                         shuffle=True, num_workers=2)

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Models
class Gen(nn.Module):
    def __init__(self):
        super(Gen, self).__init__()
        self.embed = nn.Embedding(80, 49)  # from 0 to 79 years old, 7 * 7 image
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
        self.deocde = nn.Sequential(
            nn.ConvTranspose2d(65, 64, 3, stride=3),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 3, stride=3),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 3, stride=2),
            nn.ReLU(True),
        )

    def forward(self, x, con):
        latent_vec = self.encode(x)
        latent_vec = torch.cat((torch.reshape(self.embed(con), (1, 1, 7, 7)), latent_vec), 1)
        output = self.decode(latent_vec)
        output = output.view(3, 259, 259)
        return output

# class Age_Dis(nn.Module):

class Id_Dis(nn.Module):
    def __init__(self):
        super(Id_Dis, self).__init__()

# class Age_Reg(nn.Module):




# for (img, age, name) in dataset:
#     print(age, name)


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

# # out = out.resize_(3, 259, 259)

# unloader = transforms.ToPILImage()
# out = unloader(out)
# out.save('res.jpg')

# --------------Training---------------------
