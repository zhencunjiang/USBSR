import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F
from torch.distributions import Normal, Independent


__all__ = ['resnet50_cifar', 'resnet18_cifar', 'SimpleNet']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class Bottleneck_CIFAR(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlock_CIFAR(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_CIFAR(nn.Module):

    def __init__(self, block, layers, num_features=128, reparam=False):
        super(ResNet_CIFAR, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.num_features = num_features
        self.reparam = reparam

        self.layer_block = nn.Sequential(
            self._make_layer(block, 64, layers[0], stride=1),
            self._make_layer(block, 128, layers[1], stride=2),
            self._make_layer(block, 256, layers[2], stride=2),
            self._make_layer(block, 512, layers[2], stride=2)
        )

        self.linear = nn.Linear(512 * block.expansion, num_features)

        self.learning_head = nn.Sequential(
            nn.Linear(512 * block.expansion, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_features)
        )

    def _make_layer(self, block, planes, num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.layer_block(x)

        x = F.avg_pool2d(x, 4)

        # Feature representation, 2048
        h_out = x.view(x.size(0), -1)

        # Learning output
        z_out = self.learning_head(h_out)

        if self.reparam:
            mu, sigma = z_out[:, :self.num_features // 2], z_out[:, self.num_features // 2:]
            sigma = F.softplus(sigma) + 1e-7
            return F.normalize(h_out, dim=-1), Independent(Normal(loc=mu, scale=sigma), 1)
        else:
            return F.normalize(h_out, dim=-1), F.normalize(z_out, dim=-1)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    """ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    """ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50_cifar(num_features=128, reparam=False):
    return ResNet_CIFAR(Bottleneck_CIFAR, [3, 4, 6, 3], num_features, reparam)


def resnet18_cifar(num_features=128, reparam=False):
    return ResNet_CIFAR(BasicBlock_CIFAR, [2, 2, 2, 2], num_features, reparam)


class SimpleNet(nn.Module):
    def __init__(self, input_dim=2048, num_classes=10):
        super(SimpleNet, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.fc = nn.Linear(self.input_dim, self.num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x



import os
from collections import OrderedDict

import torch
from torch import nn as nn
from torchvision.models import vgg as vgg

VGG_PRETRAIN_PATH = "/home/ps/zhencunjiang/US_sr/SimCLR-master/SimCLR-master/vgg19-dcbb9e9d.pth"
NAMES = {
    "vgg11": [
        "conv1_1",
        "relu1_1",
        "pool1",
        "conv2_1",
        "relu2_1",
        "pool2",
        "conv3_1",
        "relu3_1",
        "conv3_2",
        "relu3_2",
        "pool3",
        "conv4_1",
        "relu4_1",
        "conv4_2",
        "relu4_2",
        "pool4",
        "conv5_1",
        "relu5_1",
        "conv5_2",
        "relu5_2",
        "pool5",
    ],
    "vgg13": [
        "conv1_1",
        "relu1_1",
        "conv1_2",
        "relu1_2",
        "pool1",
        "conv2_1",
        "relu2_1",
        "conv2_2",
        "relu2_2",
        "pool2",
        "conv3_1",
        "relu3_1",
        "conv3_2",
        "relu3_2",
        "pool3",
        "conv4_1",
        "relu4_1",
        "conv4_2",
        "relu4_2",
        "pool4",
        "conv5_1",
        "relu5_1",
        "conv5_2",
        "relu5_2",
        "pool5",
    ],
    "vgg16": [
        "conv1_1",
        "relu1_1",
        "conv1_2",
        "relu1_2",
        "pool1",
        "conv2_1",
        "relu2_1",
        "conv2_2",
        "relu2_2",
        "pool2",
        "conv3_1",
        "relu3_1",
        "conv3_2",
        "relu3_2",
        "conv3_3",
        "relu3_3",
        "pool3",
        "conv4_1",
        "relu4_1",
        "conv4_2",
        "relu4_2",
        "conv4_3",
        "relu4_3",
        "pool4",
        "conv5_1",
        "relu5_1",
        "conv5_2",
        "relu5_2",
        "conv5_3",
        "relu5_3",
        "pool5",
    ],
    "vgg19": [
        "conv1_1",
        "relu1_1",
        "conv1_2",
        "relu1_2",
        "pool1",
        "conv2_1",
        "relu2_1",
        "conv2_2",
        "relu2_2",
        "pool2",
        "conv3_1",
        "relu3_1",
        "conv3_2",
        "relu3_2",
        "conv3_3",
        "relu3_3",
        "conv3_4",
        "relu3_4",
        "pool3",
        "conv4_1",
        "relu4_1",
        "conv4_2",
        "relu4_2",
        "conv4_3",
        "relu4_3",
        "conv4_4",
        "relu4_4",
        "pool4",
        "conv5_1",
        "relu5_1",
        "conv5_2",
        "relu5_2",
        "conv5_3",
        "relu5_3",
        "conv5_4",
        "relu5_4",
        "pool5",
    ],
}


def insert_bn(names):
    """Insert bn layer after each conv.
    Args:
        names (list): The list of layer names.
    Returns:
        list: The list of layer names with bn layers.
    """
    names_bn = []
    for name in names:
        names_bn.append(name)
        if "conv" in name:
            position = name.replace("conv", "")
            names_bn.append("bn" + position)
    return names_bn


class VGGFeatureExtractor(nn.Module):
    """VGG network for feature extraction.
    In this implementation, we allow users to choose whether use normalization
    in the input feature and the type of vgg network. Note that the pretrained
    path must fit the vgg type.
    Args:
        layer_name_list (list[str]): Forward function returns the corresponding
            features according to the layer_name_list.
            Example: {'relu1_1', 'relu2_1', 'relu3_1'}.
        vgg_type (str): Set the type of vgg network. Default: 'vgg19'.
        use_input_norm (bool): If True, normalize the input image. Importantly,
            the input feature must in the range [0, 1]. Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        requires_grad (bool): If true, the parameters of VGG network will be
            optimized. Default: False.
        remove_pooling (bool): If true, the max pooling operations in VGG net
            will be removed. Default: False.
        pooling_stride (int): The stride of max pooling operation. Default: 2.
    """

    def __init__(
        self,
        layer_name_list,
        vgg_type="vgg19",
        use_input_norm=True,
        range_norm=False,
        requires_grad=False,
        remove_pooling=False,
        pooling_stride=2,
    ):
        super(VGGFeatureExtractor, self).__init__()

        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        self.names = NAMES[vgg_type.replace("_bn", "")]
        if "bn" in vgg_type:
            self.names = insert_bn(self.names)

        # only borrow layers that will be used to avoid unused params
        max_idx = 0
        for v in layer_name_list:
            idx = self.names.index(v)
            if idx > max_idx:
                max_idx = idx

        if os.path.exists(VGG_PRETRAIN_PATH):
            vgg_net = getattr(vgg, vgg_type)(pretrained=False)
            state_dict = torch.load(
                VGG_PRETRAIN_PATH, map_location=lambda storage, loc: storage
            )
            vgg_net.load_state_dict(state_dict)
        else:
            vgg_net = getattr(vgg, vgg_type)(pretrained=True)

        features = vgg_net.features[: max_idx + 1]

        modified_net = OrderedDict()
        for k, v in zip(self.names, features):
            if "pool" in k:
                # if remove_pooling is true, pooling operation will be removed
                if remove_pooling:
                    continue
                else:
                    # in some cases, we may want to change the default stride
                    modified_net[k] = nn.MaxPool2d(kernel_size=2, stride=pooling_stride)
            else:
                modified_net[k] = v

        self.vgg_net = nn.Sequential(modified_net)

        if not requires_grad:
            self.vgg_net.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.vgg_net.train()
            for param in self.parameters():
                param.requires_grad = True

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer(
                "mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            )
            # the std is for image with range [0, 1]
            self.register_buffer(
                "std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            )

    def forward(self, x):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """
        if self.range_norm:
            x = (x + 1) / 2
        if self.use_input_norm:
            x = (x - self.mean) / self.std

        output = {}
        for key, layer in self.vgg_net._modules.items():
            x = layer(x)
            if key in self.layer_name_list:
                output[key] = x.clone()

        return output

import  torchvision

class vgg19(nn.Module):
    def __init__(self,pretrained=False):
       super().__init__()
       self.model= torchvision.models.vgg19()
       state_dict = torch.load(
           VGG_PRETRAIN_PATH, map_location=lambda storage, loc: storage
       )
       self.model.load_state_dict(state_dict)

       self.model.classifier = nn.Linear(25088, 128)
    def forward(self,x):
        x=self.model(x)
        return x

if __name__ == '__main__':
    # Forward pass


    x = torch.randn(1, 3,256, 256).cuda()  # Input tensor with shape (batch_size, channels, height, width)
    # y=x.view(x.shape[0], -1)
    # print(y.shape)
    # Model
    # model =VGGFeatureExtractor(layer_name_list=list({'pool5': 1})).cuda()
    model=vgg19().cuda()

    # Forward pass
    out = model(x)  # Output tensor from the model
    print(out.shape)  # Print the shape of the output tensor