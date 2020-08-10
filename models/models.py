import torch
import torch.nn as nn

'''
encoder: conv + flatten + fc
decoder: embed + fc + deconv
age regressor: vgg19 num_classes=1
discriminator for image: judge real image or not
discriminator for Z: judge z follows distribution or not

regression: none
'''


class Encoder(nn.Module):  # 3 * 224 * 224 -> 100
    def __init__(self):
        super(Encoder, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            # nn.BatchNorm2d(64),
            nn.ReLU(True),  # 64 112 112
            nn.Conv2d(64, 128, 4, 2, 1),
            # nn.BatchNorm2d(128),
            nn.ReLU(True),  # 128 56 56
            nn.Conv2d(128, 256, 4, 2, 1),
            # nn.BatchNorm2d(256),
            nn.ReLU(True),  # 256 28 28
            nn.Conv2d(256, 512, 4, 2, 1),
            # nn.BatchNorm2d(512),
            nn.ReLU(True),  # 512 14 14
            nn.Conv2d(512, 512, 4, 2, 1),
            # nn.BatchNorm2d(512),
            nn.ReLU(True),  # 512 7 7
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 100),  # z
        )

    def forward(self, x):
        return self.encode(x)


class Generator(nn.Module):  # 100 + 100 -> 3 * 224 * 224
    def __init__(self):
        super(Generator, self).__init__()
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
            # nn.ConvTranspose2d(64, 64, 4, 2, 1),      # 64 224 224
            # nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),      # 3 224 224
            nn.Tanh(),
        )

    def forward(self, vector, label):
        label = self.embed(label)
        vector = torch.cat((vector, label), 1)
        output = self.decode_fc(vector).view(-1, 1024, 7, 7)
        output = self.decode_deconv(output)
        return output


class AgeRegressor(nn.Module):  # 3 * 224 * 224 -> 1(age)
    def __init__(self):
        super(AgeRegressor, self).__init__()
        self.model = nn.Sequential(
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
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1),
        )

    def forward(self, img):
        return self.model(img)


class ImageDiscriminator(nn.Module):  # 3 * 224 * 224 -> 1(real or not)
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
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
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        return self.model(img)


class VectorDiscriminator(nn.Module):  # 100 -> 1
    def __init__(self):
        super(VectorDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, vector):
        return self.model(vector)


class FinalAgeRegressor(nn.Module):
    def __init__(self):
        super(FinalAgeRegressor, self).__init__()
        self.model = nn.Sequential()

    def forward(self, x):
        return x
