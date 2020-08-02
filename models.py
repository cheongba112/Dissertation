import torch
import torch.nn as nn

from torchvision.models import vgg19


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
            nn.Linear(64 * 9 * 9, 1024),
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

class Age_Reg(nn.Module):
    def __init__(self):
        super(Age_Reg, self).__init__()
        self.model = nn.Sequential()

    def forward(self, x):
        return x        
