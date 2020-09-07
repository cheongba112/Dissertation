import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(10, 10, 64),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 5, 2, 2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Conv2d(64, 128, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(256, 512, 5, 2, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512*8*8, 50),
            nn.Sigmoid(),
        )
    def forward(self, img, label):
        # label = label.view(label.size()[0], 10, 1, 1)
        # label = self.deconv(label)
        img = self.conv(img)
        # vector = torch.cat((img, label), 1)
        # vector = self.conv(vector)
        output = self.fc(img)
        return output


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.decode_fc = nn.Sequential(
            nn.Linear(50 + 10, 1024),
            nn.ReLU(),
            nn.Linear(1024, 8 * 8 * 1024),
            nn.ReLU(),
        )
        self.decode_deconv = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, 1, 1),
            nn.Tanh(),
        )
    def forward(self, vector, label):
        vector = torch.cat((vector, label), 1)
        output = self.decode_fc(vector).view(-1, 1024, 8, 8)
        output = self.decode_deconv(output)
        return output


class VectorDiscriminator(nn.Module):
    def __init__(self):
        super(VectorDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(50, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
    def forward(self, vector):
        return self.model(vector)


class AgeDiscriminator(nn.Module):
    def __init__(self):
        super(AgeDiscriminator, self).__init__()
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(10, 10, 64),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2),
            nn.ReLU(),
        )
        self.model = nn.Sequential(
            nn.Conv2d(26, 32, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, 2, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )
    def forward(self, img, label):
        label = label.view(label.size()[0], 10, 1, 1)
        label = self.deconv1(label)
        img = self.conv2(img)
        mid = torch.cat((label, img), 1)
        output = self.model(mid)
        return output


class AgeClassifier(nn.Module):
    def __init__(self):
        super(AgeClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(512*8*8, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 10),
            # nn.LogSoftmax(dim=1),
        )
    def forward(self, img):
        return self.model(img)


class AgeRegressor(nn.Module):
    def __init__(self):
        super(AgeRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.model(x)
