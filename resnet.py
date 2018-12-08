from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import numpy as np

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def upsample(tensor, scale_factor=2):
    """upsampling by factor"""
    return F.interpolate(tensor, scale_factor=scale_factor, mode='bilinear', align_corners=True)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, H=160, W=608):
        super(ResNet, self).__init__()

        self.H = H
        self.W = W

        self.inplanes = 64
        
        # first conv layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # stage 1-4
        self.layer1 = self.make_stage(block, 64, layers[0])
        self.score_pool1 = conv1x1(256, 1)
        self.layer2 = self.make_stage(block, 128, layers[1], stride=2)
        self.score_pool2 = conv1x1(512, 1)
        self.layer3 = self.make_stage(block, 256, layers[2], stride=2)
        self.score_pool3 = conv1x1(1024, 1)
        self.layer4 = self.make_stage(block, 512, layers[3], stride=2)
        self.score_pool4 = conv1x1(2048, 1)

        # learnable vertical flow
        self.v_flow = nn.Parameter(torch.zeros((1, H, W)))

        # coordinate grid, normalized to [-1, 1] to fit into grid_sample
        coord_x = np.tile(range(W), (H, 1)) / ((W-1)/2) - 1
        coord_y = np.tile(range(H), (W, 1)).T / ((H-1)/2) - 1
        grid = np.stack([coord_x, coord_y])
        grid = np.transpose(grid, [1,2,0])
        self.grid = nn.Parameter(torch.Tensor(grid), requires_grad=False)

        # sobel x filter
        edge_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        self.D_edge_x = nn.Parameter(edge_x.view((1,1,3,3)), requires_grad=False)
        self.edge_x = nn.Parameter(torch.cat([self.D_edge_x]*3, dim=1), requires_grad=False)

        # sobel y filter
        edge_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        self.D_edge_y = nn.Parameter(edge_y.view((1,1,3,3)), requires_grad=False)
        self.edge_y = nn.Parameter(torch.cat([self.D_edge_y]*3, dim=1), requires_grad=False)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        #nn.init.constant_(self.score_pool1.weight, 0)
        #nn.init.constant_(self.score_pool2.weight, 0)
        #nn.init.constant_(self.score_pool3.weight, 0)
        #nn.init.constant_(self.score_pool4.weight, 0)


    def make_stage(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, img_left, img_right):
        
        num_pairs = img_left.size()[0]

        x = img_left

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        score_pool1 = self.score_pool1(x)
        x = self.layer2(x)
        score_pool2 = self.score_pool2(x)
        x = self.layer3(x)
        score_pool3 = self.score_pool3(x)
        x = self.layer4(x)
        score_pool4 = self.score_pool4(x)

        fuse_pool3 = upsample(score_pool4) + score_pool3
        fuse_pool2 = upsample(score_pool3) + score_pool2
        fuse_pool1 = upsample(score_pool2) + score_pool1

        # disparity, D(x), aka. horizontal flow
        disparity = upsample(fuse_pool1, scale_factor=4)
        # normalize
        h_flow = disparity

        # vertical flow
        v = h_flow * self.v_flow

        # warping transformation
        trans = torch.cat([h_flow, v], dim=1)

        grid_warp = self.grid + trans.permute(0,2,3,1)

        # back warping
        img_warp = F.grid_sample(img_right, grid_warp)

        # reconstruction loss
        loss_recon = torch.sum((img_warp - img_left)**2) / num_pairs

        # gradient of left image and disparity
        g_x = F.conv2d(img_left, self.edge_x)
        D_g_x = F.conv2d(h_flow, self.D_edge_x)
        g_y = F.conv2d(img_left, self.edge_y)
        D_g_y = F.conv2d(h_flow, self.D_edge_y)
        
        # smoothness loss
        loss_smooth = torch.sum((g_x * D_g_x)**2) / num_pairs + torch.sum((g_y * D_g_y)**2) / num_pairs

        return loss_recon, loss_smooth, disparity


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        model.load_state_dict(pretrained_dict, strict=False)

    nn.init.constant_(model.score_pool1.weight, 0)
    nn.init.constant_(model.score_pool2.weight, 0)
    nn.init.constant_(model.score_pool3.weight, 0)
    nn.init.constant_(model.score_pool4.weight, 0)

    return model

