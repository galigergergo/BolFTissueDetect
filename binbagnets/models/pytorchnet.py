import torch.nn as nn
import math
import torch
from torch.utils import model_zoo

__all__ = ['binbagnet_small', 'binbagnet9', 'binbagnet17', 'binbagnet33']

model_urls = {
            'bagnet9': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet8-34f4ccd2.pth.tar',
            'bagnet17': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet16-105524de.pth.tar',
            'bagnet33': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet32-2ddd53ed.pth.tar',
                            }


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 kernel_size=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size,
                               stride=stride, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, **kwargs):
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
        
        if residual.size(-1) != out.size(-1):
            diff = residual.size(-1) - out.size(-1)
            residual = residual[:, :, :-diff, :-diff]
        
        out += residual
        out = self.relu(out)

        return out


class BagNetBinarySmall(nn.Module):

    def __init__(self):
        super(BagNetBinarySmall, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(1, stride=1)
        self.fc = nn.Linear(64, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        x = nn.AvgPool2d(x.size()[2], stride=1)(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class BagNetBinary(nn.Module):

    def __init__(self, block, layers, strides=[1, 2, 2, 2],
                 kernel3=[0, 0, 0, 0], avg_pool=True):
        self.inplanes = 64
        super(BagNetBinary, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0],
                                       stride=strides[0], kernel3=kernel3[0],
                                       prefix='layer1')
        self.layer2 = self._make_layer(block, 128, layers[1],
                                       stride=strides[1], kernel3=kernel3[1],
                                       prefix='layer2')
        self.layer3 = self._make_layer(block, 256, layers[2],
                                       stride=strides[2], kernel3=kernel3[2],
                                       prefix='layer3')
        self.layer4 = self._make_layer(block, 512, layers[3],
                                       stride=strides[3], kernel3=kernel3[3],
                                       prefix='layer4')
        self.avgpool = nn.AvgPool2d(1, stride=1)
        self.fc = nn.Linear(512 * block.expansion, 1)
        self.avg_pool = avg_pool
        self.block = block

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, kernel3=0,
                    prefix=''):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        kernel = 1 if kernel3 == 0 else 3
        layers.append(block(self.inplanes, planes, stride, downsample,
                            kernel_size=kernel))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            kernel = 1 if kernel3 <= i else 3
            layers.append(block(self.inplanes, planes, kernel_size=kernel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.avg_pool:
            x = nn.AvgPool2d(x.size()[2], stride=1)(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.fc(x)

        return x


def binbagnet_small(**kwargs):
    """Constructs a Bagnet test model for binary classification.
    """
    model = BagNetBinarySmall(**kwargs)
    return model


def binbagnet9(pretrained=False, strides=[2, 2, 2, 1], **kwargs):
    """Constructs a Bagnet-9 model for binary classification.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BagNetBinary(Bottleneck, [3, 4, 6, 3], strides=strides,
                         kernel3=[1, 1, 0, 0], **kwargs)
    if pretrained:
        pretrained_state = model_zoo.load_url(model_urls['bagnet9'])
        pretrained_state['fc.weight'] =\
            torch.mean(pretrained_state['fc.weight'], 0, True)
        pretrained_state['fc.bias'] =\
            torch.mean(pretrained_state['fc.bias'], 0, True)
        model.load_state_dict(pretrained_state)
    return model


def binbagnet17(pretrained=False, strides=[2, 2, 2, 1], **kwargs):
    """Constructs a Bagnet-17 model for binary classification.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BagNetBinary(Bottleneck, [3, 4, 6, 3], strides=strides,
                         kernel3=[1, 1, 1, 0], **kwargs)
    if pretrained:
        pretrained_state = model_zoo.load_url(model_urls['bagnet17'])
        pretrained_state['fc.weight'] =\
            torch.mean(pretrained_state['fc.weight'], 0, True)
        pretrained_state['fc.bias'] =\
            torch.mean(pretrained_state['fc.bias'], 0, True)
        model.load_state_dict(pretrained_state)
    return model


def binbagnet33(pretrained=False, strides=[2, 2, 2, 1], **kwargs):
    """Constructs a Bagnet-33 model for binary classification.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BagNetBinary(Bottleneck, [3, 4, 6, 3], strides=strides,
                         kernel3=[1, 1, 1, 1], **kwargs)
    if pretrained:
        pretrained_state = model_zoo.load_url(model_urls['bagnet33'])
        pretrained_state['fc.weight'] =\
            torch.mean(pretrained_state['fc.weight'], 0, True)
        pretrained_state['fc.bias'] =\
            torch.mean(pretrained_state['fc.bias'], 0, True)
        model.load_state_dict(pretrained_state)
    return model
