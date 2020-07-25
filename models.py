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
        self.embed = nn.Embedding(100, 100)
        self.encode = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d()
        )
        self.deocde = nn.Sequential(

        )

    def forward(self, x, con):
        latent_vec = self.encode(x)
        latent_vec = torch.cat((self.embed(con), x), -1)
        output = self.decode(latent_vec)
        return output

# class Age_Dis(nn.Module):

# class Id_Dis(nn.Module):

# class Age_Reg(nn.Module):




# for (img, age, name) in dataset:
#     print(age, name)



