'''
This py file is customised for importing CACD2000 dataset
Inspired by https://blog.csdn.net/woshicao11/article/details/78318156
'''
import torch
from torch.utils.data import Dataset

import os, re, pprint

from PIL import Image
from torchvision import transforms


def get_img_list(root):
    img_list = []
    pattern = re.compile('([0-9]+)(\_)(.+)(\_[0-9]+)')

    for img_name in os.listdir(root):
        img_match = re.search(pattern, img_name)
        
        img_path = os.path.join(root, img_name)
        img_age  = img_match.group(1)
        img_id   = img_match.group(3)
        
        img_list.append([img_path, img_age, img_id])
    
    return img_list


# pprint.pprint(get_img_list('./14'))


class CACD_Dataset(Dataset):
    def __init__(self, root):
        self.img_list = get_img_list(root)

    def __getitem__(self, idx):
        img_path, img_age, img_id = self.img_list[idx]

        img = Image.open(img_path)
        trans = transforms.Compose([
            transforms.Resize(250),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # from 0,1 to -1,1
            ])
        img = trans(img)

        return img, int(img_age), img_id

    def __len__(self):
        return len(self.img_list)


# cacd = CACD_Dataset('./14')
# pprint.pprint(cacd[0][0].size())
