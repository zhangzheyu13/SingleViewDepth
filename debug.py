from __future__ import division
import numpy as np
import torch
from resnet import resnet50
from PIL import Image
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

# load data
H = 160
W = 608
data_transform = transforms.Compose([
        transforms.Resize(size=(H, W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

left_file = open('train_left.txt')
imgs = []
for line in left_file.readlines():
    img = Image.open(line[:-2])
    img = data_transform(img)
    imgs.append(img)
img_left = torch.stack(imgs)
left_file.close()

# load the same images, see if it can learn zero disparity
right_file = open('train_left.txt')
imgs = []
for line in right_file.readlines():
    img = Image.open(line[:-2])
    img = data_transform(img)
    imgs.append(img)
img_right = torch.stack(imgs)

net = resnet50(pretrained=False)
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)

num_epoch = 1000
for i in range(num_epoch):

    optimizer.zero_grad()

    loss_recon, loss_smooth = net(img_left, img_right)

    loss = loss_recon + loss_smooth
    loss.backward()
    optimizer.step()

    print('epoch [{}], loss recon {}, loss smooth {}'.format(i, loss_recon.item(), loss_smooth.item()))
