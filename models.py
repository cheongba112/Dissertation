import os, numpy, argparse
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import vgg19

from PIL import Image


# Create the dataset
dataset = dset.ImageFolder(root='CACD2000',
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), 
                                                    (0.5, 0.5, 0.5)),
                           ]))

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=100,
                                         shuffle=True, num_workers=2)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Models
class Gen(nn.Module):
    def __init__(self):
        super(Gen, self).__init__()
        self.embed = 
        self.encode = nn.Sequential(
            nn.Conv2d(3, )
        )
        self.deocde = nn.Sequential(

        )

    def forward(self, x, con):
        latent_vec = self.encode(x)
        latent_vec = torch.cat((self.embed(con), x), -1)
        output = self.decode(latent_vec)
        output = output.view(, )
        return output

# class Age_Dis(nn.Module):

# class Id_Dis(nn.Module):

# class Age_Reg(nn.Module):

