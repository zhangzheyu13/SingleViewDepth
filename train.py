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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

# load data
H = 160
W = 608
data_transform = transforms.Compose([
        transforms.Resize(size=(H, W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

depth_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=(H, W)),
        transforms.ToTensor()
    ])

kitti_train = KittiDataset(data_transform=data_transform)
train_dataloader = DataLoader(kitti_train, batch_size=16, shuffle=True, **kwargs)
#kitti_test = KittiDataset(root_dir='./images/test', train=False, data_transform=data_transform, depth_transform=depth_transform)
#test_dataloader = DataLoader(kitti_test, batch_size=1, shuffle=True, **kwargs)

net = resnet50(pretrained=False).to(device)
#print(net)
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)

# start training
num_epoch = 100
for e in range(num_epoch):

    # training
    net.train()
    for i, batch in enumerate(train_dataloader):
        img_left = batch['img_left'].to(device)
        img_right = batch['img_right'].to(device)

        optimizer.zero_grad()

        loss_recon, loss_smooth, _ = net(img_left, img_right)

        loss = loss_recon + 0.01 * loss_smooth
        loss.backward()
        optimizer.step()

        print('epoch [{}], iter [{}], loss recon {}, loss smooth {}'.format(e, i, loss_recon.item(), loss_smooth.item()))

    # validation
    # how to evaluate the model?
