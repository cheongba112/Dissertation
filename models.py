import torch
import torch.nn as nn


class Encoder(nn.Module):  # 3 * 224 * 224 -> 50
    def __init__(self):
        super(Encoder, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),  # 64 112 112
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(),  # 64 56 56
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),  # 128 28 28
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),  # 256 14 14
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.ReLU(),  # 512 7 7
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 50),
            nn.Sigmoid()   # z, 50, range between (0, 1)
        )

    def forward(self, x):
        return self.encode(x)


class Generator(nn.Module):  # 50 + 100 -> 3 * 224 * 224
    def __init__(self):
        super(Generator, self).__init__()
        self.decode_fc = nn.Sequential(
            nn.Linear(10 + 50, 7 * 7 * 1024),
            nn.ReLU(),
        )
        self.decode_deconv = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),  # 512 14 14
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),   # 256 28 28
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),   # 128 56 56
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),    # 64 112 112
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),     # 64 224 224
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, 1, 1),      # 3 224 224
            nn.Tanh(),
        )

    def forward(self, vector, label):
        vector = torch.cat((vector, label), 1)
        output = self.decode_fc(vector).view(-1, 1024, 7, 7)
        output = self.decode_deconv(output)
        return output


class AgeRegressor(nn.Module):  # 3 * 224 * 224 -> 1(age)
    def __init__(self):
        super(AgeRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1),  # 16 112 112
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1),  # 32 56 56
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 64 28 28
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 128 14 14
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),  # 256 7 7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )
    
    def forward(self, img):
        return self.model(img)


class ImageDiscriminator(nn.Module):  # 3 * 224 * 224 -> 1(real or not)
    def __init__(self):
        super(ImageDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1),  # 16 112 112
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1),  # 32 56 56
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 64 28 28
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 128 14 14
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),  # 256 7 7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        return self.model(img)


class VectorDiscriminator(nn.Module):  # 50 -> 1
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


class FinalAgeRegressor(nn.Module):  # 7 * 3 * 224 * 224 -> 1(age)
    def __init__(self):
        super(FinalAgeRegressor, self).__init__()
        self.model = nn.Sequential(
            
        )

    def forward(self, x):
        return x
