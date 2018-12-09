from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from read_depth import depth_read

class KittiDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir='../images/train', train=True, use_depth=False, data_transform=None, depth_transform=None, H=320, W=1216, num_test=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.train = train
        self.use_depth = use_depth
        self.data_transform = data_transform
        self.depth_transform = depth_transform
        self.H = H
        self.W = W
        self.num_test = num_test

        self.left_img_list = []
        self.right_img_list = []
        self.left_depth_list = []
        # scan the file paths
        for d in os.listdir(root_dir):
            sub_dir = os.path.join(root_dir, d)
            for sd in os.listdir(sub_dir):
                left_img_dir = os.path.join(sub_dir, sd, 'image_02', 'data')
                for img_name in os.listdir(left_img_dir):
                    if self.use_depth:
                        depth_dir = os.path.join('../images/test_depth', sd, 'proj_depth', 'groundtruth', 'image_02', img_name)
                        if not os.path.isfile(depth_dir):
                            continue
                        self.left_depth_list.append(depth_dir)
                    self.left_img_list.append(os.path.join(left_img_dir, img_name))

                right_img_dir = os.path.join(sub_dir, sd, 'image_03', 'data')
                for img_name in os.listdir(right_img_dir):
                    if self.use_depth:
                        depth_dir = os.path.join('../images/test_depth', sd, 'proj_depth', 'groundtruth', 'image_02', img_name)
                        if not os.path.isfile(depth_dir):
                            continue
                    self.right_img_list.append(os.path.join(right_img_dir, img_name))

        assert len(self.left_img_list) == len(self.right_img_list)

        self.iterator = 0


    def __len__(self):
        if not self.train:
            return self.num_test
        return len(self.left_img_list)


    def __getitem__(self, idx):
        img_left = Image.open(self.left_img_list[idx])
        img_right = Image.open(self.right_img_list[idx])

        if self.data_transform:
            img_left = self.data_transform(img_left)
            img_right = self.data_transform(img_right)

        if self.use_depth:
            depth_left = depth_read(self.left_depth_list[idx])
            if self.depth_transform:
                depth_left = self.depth_transform(depth_left)

            sample = {'img_left': img_left, 'img_right': img_right, 'depth_left': depth_left}
        else:
            sample = {'img_left': img_left, 'img_right': img_right, 'img_left_dir': self.left_img_list[idx]}

        return sample
