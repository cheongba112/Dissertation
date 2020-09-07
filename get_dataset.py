import torch
from torch.utils.data import Dataset

import os, re, pprint, random

from PIL import Image
from torchvision import transforms


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
    pattern = re.compile('([0-9]+)(\_)(.+)')
    # img_dict = {}
    img_list = []

    for img_name in files:
        img_match = re.search(pattern, img_name)
        
        img_path = os.path.join(root, img_name)
        img_age  = img_match.group(1)
        
        img_list.append([img_path, img_age])

    return img_list


class get_dataset(Dataset):
    def __init__(self, root):
        self.img_list = get_img_list(root)

    def __getitem__(self, idx):
        src_img_path, img_age = self.img_list[idx]
        return open_image(src_img_path), int(img_age)

    def __len__(self):
        return len(self.img_list)