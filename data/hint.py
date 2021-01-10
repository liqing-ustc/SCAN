"""
This code is based on the Torchvision repository, which was licensed under the BSD 3-Clause.
"""
import os
import pickle
import sys
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from utils.mypath import MyPath
import json
import sys
sys.path.append('../CLL-NeSy/data')
from domain import SYMBOLS, SYM2ID
from collections import Counter
import random

from torchvision import transforms
def pad_image(img, desired_size, fill=0):
    delta_w = desired_size - img.size[0]
    delta_h = desired_size - img.size[1]
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    new_img = ImageOps.expand(img, padding, fill)
    return new_img

class HINT(Dataset):
    """ The HINT dataset
    Args:
        root (string): Root directory of dataset where directory
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, root=MyPath.db_root_dir('hint'), split='train', transform=None):

        super(HINT, self).__init__()
        self.root = root
        self.img_dir = root + 'symbol_images/'
        self.transform = transform
        self.classes = SYMBOLS

        # split = 'train'
        self.split = split

        dataset = json.load(open(root + 'expr_%s.json'%split))
        # dataset = [x for x in dataset if len(x['expr']) <= 15]
        dataset = [(x,SYM2ID(y)) for sample in dataset for x, y in zip(sample['img_paths'], sample['expr'])]
        label2data = {i:[] for i in range(len(self.classes))}
        for img, label in dataset:
            label2data[label].append((img, label))
        dataset = []
        random.seed(777)
        n_sample_per_class = 5000 if split == 'train' else 500
        for label, data in label2data.items():
            dataset.extend(random.choices(data, k=n_sample_per_class))
        random.shuffle(dataset)
        print(dataset[:10])

        print(sorted(Counter([x[1] for x in dataset]).items()))
        self.dataset = dataset

    def __getitem__(self, index):
        sample = self.dataset[index]
        img_path, target = sample
        img = Image.open(self.img_dir+img_path).convert('L')
        img = ImageOps.invert(img)
        img = pad_image(img, 60)
        img = transforms.functional.resize(img, 40)
        img_size = img.size
        class_name = self.classes[target]        

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {'im_size': img_size, 'index': index, 'class_name': class_name}}
        
        return out
        
    def __len__(self):
        return len(self.dataset)

    def get_image(self, index):
        sample = self.dataset[index]
        img_path, target = sample
        img = Image.open(self.img_dir+img_path).convert('L')
        img = ImageOps.invert(img)
        return img

    def extra_repr(self):
        return "Split: {}".format(self.split)