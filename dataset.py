from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

class KittiDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir='./images/train', transform=None, H=160, W=608):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.H = H
        self.W = W

        self.left_img_list = []
        self.right_img_list = []
        # scan the file paths
        for d in os.listdir(root_dir):
            sub_dir = os.path.join(root_dir, d)
            for sd in os.listdir(sub_dir):
                left_img_dir = os.path.join(sub_dir, sd, 'image_02', 'data')
                for img_name in os.listdir(left_img_dir):
                    self.left_img_list.append(os.path.join(left_img_dir, img_name))

                right_img_dir = os.path.join(sub_dir, sd, 'image_03', 'data')
                for img_name in os.listdir(right_img_dir):
                    self.right_img_list.append(os.path.join(right_img_dir, img_name))

        assert len(self.left_img_list) == len(self.right_img_list)

        self.iterator = 0


    def __len__(self):
        return len(self.left_img_list)


    def __getitem__(self, idx):
        img_left = Image.open(self.left_img_list[idx])
        img_right = Image.open(self.right_img_list[idx])

        if self.transform:
            img_left = self.transform(img_left)
            img_right = self.transform(img_right)

        sample = {'img_left': img_left, 'img_right': img_right}

        return sample
