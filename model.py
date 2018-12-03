import torch.nn as nn
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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

        print('x: {}'.format(x.size()))
        print('out: {}'.format(out.size()))

        if self.downsample is not None:
            residual = self.downsample(x)

        print('residual: {}'.format(residual.size()))

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.inplanes = 128

        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        self.relu0 = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.score_pool1 = conv1x1(128, 1)

        self.stage1 = self.make_stage(block, 32, layers[0])
        self.score_pool2 = conv1x1(128, 1)

        self.stage2 = self.make_stage(block, 64, layers[1], stride=2)
        self.score_pool3 = conv1x1(256, 1)

        self.stage3 = self.make_stage(block, 128, layers[2], stride=2)
        self.score_pool4 = conv1x1(512, 1)

        self.stage4 = self.make_stage(block, 256, layers[3], stride=2)
        self.score_pool5 = conv1x1(1024, 1)

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_stage(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
            stride = 1

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.pool0(x)

        print('pool0: {}'.format(x.size()))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        score_pool1 = self.score_pool1(x)

        print('pool1: {}'.format(x.size()))

        x = self.stage1(x)
        score_pool2 = self.score_pool2(x)
        x = self.stage2(x)
        score_pool3 = self.score_pool3(x)
        x = self.stage3(x)
        score_pool4 = self.score_pool4(x)
        x = self.stage4(x)
        score_pool5 = self.score_pool5(x)

        fuse_pool4 = self.upsample(score_pool5) + score_pool4
        fuse_pool3 = self.upsample(score_pool4) + score_pool3
        fuse_pool2 = self.upsample(score_pool3) + score_pool2
        fuse_pool1 = self.upsample(score_pool2) + score_pool1

        disparity = upsample(fuse_pool1)

        return disparity



def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model
