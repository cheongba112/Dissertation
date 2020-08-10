import torch
import torch.nn as nn

from torchvision.models import vgg19, resnet34

'''
G:    conv encode -> age embed -> cat -> fc 1024 -> deconv decode
ageD: res34 encode -> age embed -> cat -> fc 1
idD:  res34 encode * 2 -> cat -> fc 1
regression: none
'''


class Gen(nn.Module):
    def __init__(self):
        super(Gen, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            # nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2),  # 64 112 112
            nn.Conv2d(64, 128, 3, 1, 1),
            # nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2),  # 128 56 56
            nn.Conv2d(128, 256, 3, 1, 1),
            # nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2),  # 256 28 28
            nn.Conv2d(256, 512, 3, 1, 1),
            # nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2),  # 512 14 14
            nn.Conv2d(512, 512, 3, 1, 1),
            # nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2),  # 512 7 7
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 100),  # z
        )
        self.embed = nn.Embedding(100, 100)      # classes, vector length = 100
        self.embed.weight.data = torch.eye(100)  # one-hot encoding
        self.decode_fc = nn.Sequential(
            nn.Linear(200, 7 * 7 * 1024),
            nn.ReLU(True),
        )
        self.decode_deconv = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),  # 512 14 14
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),   # 256 28 28
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),   # 128 56 56
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),    # 64 112 112
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),      # 3 224 224
            nn.Tanh(),
        )

    def forward(self, img, label):
        vector = self.encode(img)
        label = self.embed(label)
        vector = torch.cat((vector, label), 1)
        output = self.decode_fc(vector).view(-1, 1024, 7, 7)
        output = self.decode_deconv(output)
        return output

class AgeD(nn.Module):
    def __init__(self):
        super(AgeD, self).__init__()
        self.encode = resnet34()
        self.embed = nn.Embedding(100, 100)
        self.embed.weight.data = torch.eye(100)
        self.fc = nn.Sequential(
            nn.Linear(1100, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, age):
        vector = self.encode(img)
        label = self.embed(age)
        vector = torch.cat((vector, label), 1)
        output = self.fc(vector)
        return output

class IdD(nn.Module):
    def __init__(self):
        super(IdD, self).__init__()
        self.encode = resnet34()
        self.fc = nn.Sequential(
            nn.Linear(2000, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, img_1, img_2):
        vector_1 = self.encode(img_1)
        vector_2 = self.encode(img_2)
        vector = torch.cat((vector_1, vector_2), 1)
        output = self.fc(vector)
        return output

class AgeR(nn.Module):
    def __init__(self):
        super(AgeR, self).__init__()
        self.model = nn.Sequential()

    def forward(self, x):
        return x
