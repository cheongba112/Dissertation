'''
Inspired by https://blog.csdn.net/woshicao11/article/details/78318156
'''
import torch
from torch.utils.data import Dataset

import os, re, pprint, random

from PIL import Image
from torchvision import transforms


def open_image(img_path):
    img = Image.open(img_path)
    trans = transforms.Compose([
        transforms.Resize(224),
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


if __name__ == '__main__':
    pprint.pprint(get_img_list('./14'), width=999)

    cacd = get_dataset('./CACD2000')  # 3+ seconds
    pprint.pprint(cacd[0])
