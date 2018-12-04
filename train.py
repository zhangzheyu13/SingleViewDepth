#from model import resnet50
from __future__ import division
import numpy as np
import torch
from resnet import resnet50
import cv2
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from dataset import KittiDataset
from torch.utils.data import Dataset, DataLoader

# load data
H = 160
W = 608
data_transform = transforms.Compose([
        transforms.Resize(size=(H, W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

kitti = KittiDataset(transform=data_transform)
dataloader = DataLoader(kitti, batch_size=16, shuffle=True, num_workers=4)

net = resnet50(pretrained=False)
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)

# start training
num_epoch = 100
for e in range(num_epoch):

    # training
    for i, batch in enumerate(dataloader):

        optimizer.zero_grad()

        loss_recon, loss_smooth = net(batch['img_left'], batch['img_right'])

        loss = loss_recon + 0.01 * loss_smooth
        loss.backward()
        optimizer.step()

        print('epoch [{}], iter [{}], loss recon {}, loss smooth {}'.format(e, i, loss_recon.item(), loss_smooth.item()))

    # validation
    # how to evaluate the model?
