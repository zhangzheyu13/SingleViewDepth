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
import matplotlib.pyplot as plt
import os
import argparse
from logger import Logger
import datetime

def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-4, help="weight decay")
    parser.add_argument('--epochs', dest='epochs', type=int, default=100, help="total training epochs")
    parser.add_argument('--log_interval', dest='log_interval', type=int, default=10, help="log interval")
    parser.add_argument('--eval', action='store_true', help="evaluation mode")
    parser.add_argument('--use_depth', action='store_true', help="use depth")
    parser.add_argument('--model_dir', dest='model_dir', type=str, default='', help="model dir")
    parser.add_argument('--model_name', dest='model_name', type=str, default='my_model', help="model name")
    parser.add_argument('--num_test', dest='num_test', type=int, default=10)

    return parser.parse_args()


args = parse_arguments()

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
train_dataloader = DataLoader(kitti_train, batch_size=args.batch_size, shuffle=True, **kwargs)
kitti_test = KittiDataset(root_dir='../images/test', train=False, data_transform=data_transform, depth_transform=depth_transform, num_test=args.num_test)
test_dataloader = DataLoader(kitti_test, batch_size=1, shuffle=False, **kwargs)
print(len(kitti_test))

net = resnet50(pretrained=True).to(device)
if args.model_dir is not '':
    net.load_state_dict(torch.load(args.model_dir))

optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

log_dir = '../images/logs'
now = str(datetime.datetime.now())
log_dir = os.path.join(log_dir, now)
logger = Logger(log_dir)

# start training
best_loss = float('inf')
for e in range(args.epochs):

    # training
    if not args.eval:
        net.train()
        train_loss = []
        for i, batch in enumerate(train_dataloader):
            img_left = batch['img_left'].to(device)
            img_right = batch['img_right'].to(device)

            optimizer.zero_grad()

            loss_recon, loss_smooth, _ = net(img_left, img_right)

            loss = loss_recon + 0.01 * loss_smooth
            loss.backward()
            optimizer.step()

            if i % args.log_interval == 0:
                print('epoch [{}], iter [{}], loss recon {}, loss smooth {}'.format(e, i, loss_recon.item(), loss_smooth.item()))
            train_loss.append(loss.item())

        avg_loss = np.mean(train_loss)
        print('epoch [{}], avg loss {}'.format(e, avg_loss))
        logger.scalar_summary('train/avg_loss', avg_loss, self.cur_epoch)

        if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(net.state_dict(), '../images/models/{}_best_epoch_{}.pth'.format(model_name, e))
                print('model saved...')

    else:
        # validation
        for i, batch in enumerate(test_dataloader):
            net.eval()
            with torch.no_grad():
                img_left = batch['img_left'].to(device)
                img_right = batch['img_right'].to(device)
                if use_depth:
                    depth_left = batch['depth_left'].to(device)
                    valid_mask = (depth_left > 0)
                #print(torch.max(depth_left))
                #print(type(depth_left))
                _, _, disparity = net(img_left, img_right)

                disparity = torch.abs(disparity)
                disp = disparity.squeeze().to('cpu').numpy()
                plt.imsave(os.path.join('../images/output', 'pred_'+str(i)+'_test_output.png'), disp, cmap='plasma')
                img_left_2show = np.transpose(img_left.squeeze().to('cpu').numpy(), (1,2,0))
                plt.imsave(os.path.join('../images/output', 'img_'+str(i)+'_test_output.png'), img_left_2show)
                
                if use_depth:
                    scale = 1242
                    depth = 389.6304 / (scale * disparity)
                    #print(torch.max(depth), torch.min(depth))
                    #print(torch.max(depth_left), torch.min(depth_left))

                    #error = torch.abs(depth[valid_mask] - depth_left[valid_mask])
                    #print(depth_left)
                    #print(torch.sum((error / depth_left[valid_mask]) < 1))
                    correct = torch.sum(torch.max(depth[valid_mask] / depth_left[valid_mask], depth_left[valid_mask] / depth[valid_mask]) < 1.25*1.25*1.25).item()
                    #print(correct)
                    #print(torch.sum(valid_mask))
                    accuracy = correct / torch.sum(valid_mask).item()
                    print('epoch [{}], image [{}], accuracy {}'.format(e, i, accuracy))
            
        break
