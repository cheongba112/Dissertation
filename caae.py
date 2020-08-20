import os, random, csv, time, re, shutil
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim

from PIL import Image
from torchvision import transforms
from torchvision import utils

# split test set and valid set
def dataset_split(root_path):
    old_root = root_path
    test_root = root_path + '_test'
    valid_root = root_path + '_valid'

    if not os.path.exists(test_root):
        os.mkdir(test_root)

    if not os.path.exists(valid_root):
        os.mkdir(valid_root)

    files = os.listdir(old_root)

    tbmoved = random.sample(range(len(files)), 100)
    for i in tbmoved:
        shutil.move(os.path.join(old_root, files[i]), os.path.join(test_root, files[i]))

    tbmoved = random.sample(range(len(files)), 8)
    for i in tbmoved:
        shutil.move(os.path.join(old_root, files[i]), os.path.join(valid_root, files[i]))

# tensor to string of list
def tl(tensor):
    return str(tensor.tolist())

# weights initialise function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)

# TV loss
def TV_Loss(imgTensor, img_size=128):
    x = (imgTensor[:, :, 1:, :] - imgTensor[:, :, :img_size - 1, :]) ** 2
    y = (imgTensor[:, :, :, 1:] - imgTensor[:, :, :, :img_size - 1]) ** 2

    out = (x.mean(dim=2) + y.mean(dim=3)).mean()
    return out

# --------------------------------------------------
def open_image(img_path):
    img = Image.open(img_path)
    trans = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # from 0,1 to -1,1
    ])
    img = trans(img)
    return img

def get_img_list(root):
    files = os.listdir(root)
    pattern = re.compile('([0-9]+)(\_)(.+)(\_.+)')
    img_dict = {}
    img_list = []

    for img_name in files:
        img_match = re.search(pattern, img_name)
        
        img_path = os.path.join(root, img_name)
        img_age  = img_match.group(1)
        img_id   = img_match.group(3)

        if img_id in img_dict:
            img_dict[img_id].append([img_path, img_age])
        else:
            img_dict[img_id] = [[img_path, img_age]]

    for id in img_dict:
        sub_list = img_dict[id]
        for img in sub_list:
            idx = random.randint(0, len(sub_list) - 1)
            img_list.append([img[0], img[1], sub_list[idx][0]])

    return img_list

class get_dataset(Dataset):
    def __init__(self, root):
        self.img_list = get_img_list(root)

    def __getitem__(self, idx):
        src_img_path, img_age, tgt_img_path = self.img_list[idx]
        return open_image(src_img_path), int(img_age), open_image(tgt_img_path)

    def __len__(self):
        return len(self.img_list)
# --------------------------------------------------

val_bth = 8
res_pth = './caae_result/'

if not os.path.exists('./CACD2000_valid'):
        dataset_split('./CACD2000')

# dataset
dataset = get_dataset('./CACD2000')
validset = get_dataset('./CACD2000_valid')

# dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                         shuffle=True, num_workers=1)
validloader = torch.utils.data.DataLoader(validset, batch_size=val_bth, shuffle=False, num_workers=1)

# use cpu or gpu as device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(3, 64, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(256, 512, 5, 2, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 50),
            nn.Tanh()
        )
    def forward(self, x):
        return self.encode(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.decode_fc = nn.Sequential(
            nn.Linear(50 + 10, 8 * 8 * 1024),
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

class ImageDiscriminator(nn.Module):
    def __init__(self):
        super(ImageDiscriminator, self).__init__()
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

# model
netE    = Encoder().to(device)
netG    = Generator().to(device)
netimgD = ImageDiscriminator().to(device)
netvecD = VectorDiscriminator().to(device)

# model weights initialise
netE.apply(weights_init)
netG.apply(weights_init)
netimgD.apply(weights_init)
netvecD.apply(weights_init)

# loss function
BCE = nn.BCELoss().to(device)
L1  = nn.L1Loss().to(device)
MSE = nn.MSELoss().to(device)

# optimizer
optimE    = optim.Adam(netE.parameters(),    lr=0.0002, betas=(0.5, 0.999))
optimG    = optim.Adam(netG.parameters(),    lr=0.0002, betas=(0.5, 0.999))
optimimgD = optim.Adam(netimgD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimvecD = optim.Adam(netvecD.parameters(), lr=0.0002, betas=(0.5, 0.999))

# training
if __name__ == '__main__':
    # start = time.time()

    if not os.path.exists(res_pth):
        os.makedirs(res_pth + 'pics')

    with open(res_pth + 'train_result.csv', 'a', encoding='utf-8', newline='') as F:
        w = csv.writer(F)
        w.writerow(['epoch', 'batch', 
            'loss_imgD_T', 'loss_imgD_F', 
            'loss_vecD_T', 'loss_vecD_F', 
            'loss_l1_G', 'loss_img_G', 'loss_vec_G', 'loss_tv_G'])

    for epoch in range(30):
        print('Epoch: %d' % (epoch))
        
        for i, (src_img, src_age, _) in enumerate(dataloader):
            print('Batch: %d' % (i))
            
            # prepare batch data
            src_age = (src_age // 5) - 3
            src_age += (src_age < 0).type(torch.LongTensor)  # 14 -> -1 -> 0

            src_age = src_age.to(device)
            src_img = src_img.to(device)
            
            pri_vec = torch.FloatTensor(src_age.size()[0], 50).uniform_(-1, 1).to(device)
            
            real_label  = torch.full(src_age.size(), 1, device=device)  # warning
            false_label = torch.full(src_age.size(), 0, device=device)  # warning
            
            src_age_onehot = - torch.ones(src_age.size()[0], 10).to(device)
            for j, age in enumerate(src_age):
                src_age_onehot[j][age] = 1.

            # generate synthesized images
            syn_vec = netE(src_img)
            syn_img = netG(syn_vec, src_age_onehot)

            # train image D
            netimgD.zero_grad()
            output_img_T = netimgD(src_img, src_age_onehot).view(-1)
            output_img_F = netimgD(syn_img.detach(), src_age_onehot).view(-1)
            loss_imgD_T = BCE(output_img_T, real_label)
            loss_imgD_F = BCE(output_img_F, false_label)
            loss_imgD = loss_imgD_T + loss_imgD_F
            loss_imgD.backward()
            optimimgD.step()

            # train vector D
            netvecD.zero_grad()
            output_vec_T = netvecD(pri_vec).view(-1)
            output_vec_F = netvecD(syn_vec.detach()).view(-1)
            loss_vecD_T = BCE(output_vec_T, real_label)
            loss_vecD_F = BCE(output_vec_F, false_label)
            loss_vecD = loss_vecD_T + loss_vecD_F
            loss_vecD.backward()
            optimvecD.step()

            # train encoder and generator
            netE.zero_grad()
            netG.zero_grad()
            
            # L1 loss
            loss_l1_G = L1(syn_img, src_img)

            # image D loss
            output_img_G = netimgD(syn_img, src_age_onehot).view(-1)
            loss_img_G = BCE(output_img_G, real_label)

            # vector D loss
            output_vec_G = netvecD(syn_vec).view(-1)
            loss_vec_G = BCE(output_vec_G, real_label)

            # tv loss
            loss_tv_G = TV_Loss(syn_img)

            loss_G = loss_l1_G + 0.001 * loss_img_G + 0.05 * loss_vec_G + loss_tv_G
            loss_G.backward()

            optimE.step()
            optimG.step()

            # save loss info every 500 batchs
            if not i % 500:
                # utils.save_image(syn_img, './train_result/pics/%d_%d.jpg' % (epoch, i), normalize=True)
                with open(res_pth + 'train_result.csv', 'a', encoding='utf-8', newline='') as F:
                    writer = csv.writer(F)
                    writer.writerow([epoch, i, 
                        tl(loss_imgD_T), tl(loss_imgD_F), 
                        tl(loss_vecD_T), tl(loss_vecD_F), 
                        tl(loss_l1_G), tl(loss_img_G), tl(loss_vec_G), tl(loss_tv_G)])

        # generate test result every epoch
        for src_img, _, _ in validloader:
            src_img = src_img.to(device)
            imgs = src_img
            for i in range(7):  # src, 0, 1, 2, 3, 4, 5, 6
                syn_age_onehot = - torch.ones(val_bth, 10).to(device)
                for j in range(val_bth):
                    syn_age_onehot[j][i] = 1.
                syn_vec = netE(src_img)
                syn_img = netG(syn_vec, syn_age_onehot)
                imgs = torch.cat((imgs, syn_img), 0)
            utils.save_image(imgs, './caae_result/pics/%d.jpg' % (epoch), normalize=True)

    # print loss line charts
    with open(res_pth + 'train_result.csv', 'r') as F:
        r = csv.reader(F)
        data = [row for row in r]
        for i in range(2, len(data[0])):  # skip epoch and batch columns
            col = [row[i] for row in data]
            label = col[0]
            value = []
            for d in col[1:]:
                value.append(float(d))
            plt.plot(range(len(value)), value, '-')
            plt.ylabel(label)
            plt.savefig(res_pth + label + '.jpg')
            plt.clf()
