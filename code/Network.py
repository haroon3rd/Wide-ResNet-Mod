import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class wide_basic(nn.Module):
    def __init__(self, kernels_in, kernels_out, stride, drop_rate=0.0):
        super(wide_basic, self).__init__()

        self.bn1 = nn.BatchNorm2d(kernels_in)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(kernels_in,
                            kernels_out,
                            kernel_size=3,
                            stride=stride,
                            padding=1,
                            bias=False)
        self.dropout = nn.Dropout(p=drop_rate)
        self.bn2 = nn.BatchNorm2d(kernels_out)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(kernels_out,
                            kernels_out,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False)
        # identity connections
        self.equal = (kernels_in == kernels_out)
        self.convShortcut = None if self.equal else nn.Conv2d(kernels_in,
                                                            kernels_out,
                                                            kernel_size=1,
                                                            stride=stride,
                                                            padding=0,
                                                            bias=False)

    def forward(self, x):
        if self.equal:
            out = self.bn1(x)
            out = self.relu1(out)
            out = self.conv1(out)
            out = self.bn2(out)
            out = self.relu2(out)
        else:
            out = self.bn1(x)
            out = self.relu1(out)
            out = self.conv1(out)
            out = self.bn2(out)
            out = self.relu2(out)

        out = self.dropout(out)
        out = self.conv2(out)
        return torch.add(x, out) if self.equal else torch.add(self.convShortcut(x), out)

class wide_resnet(nn.Module):
    def __init__(self, num_layers, kernels_in, kernels_out, block, stride, drop_rate=0.0):
        super(wide_resnet, self).__init__()

        layers = []
        for i in range(num_layers):
            layers.append(block(kernels_out, kernels_out, 1, drop_rate)) if i != 0 else layers.append(block(kernels_in, kernels_out, stride, drop_rate))
        self.layer = nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)


class MyNetwork(nn.Module):
    def __init__(self, config):
        super(MyNetwork, self).__init__()
        depth_of_network = config["depth"]
        drop_rate = config["drop_rate"]
        num_of_classes = config["num_classes"]
        widen_factor = config["widen_factor"]

        num_features = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        n = (depth_of_network - 4) // 6
        block = wide_basic

        self.conv1 = nn.Conv2d(3, num_features[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = wide_resnet(n, num_features[0], num_features[1], block, 1, drop_rate)
        self.layer2 = wide_resnet(n, num_features[1], num_features[2], block, 2, drop_rate)
        self.layer3 = wide_resnet(n, num_features[2], num_features[3], block, 2, drop_rate)
        self.bn1 = nn.BatchNorm2d(num_features[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(num_features[3], num_of_classes)
        self.num_features = num_features[3]
        self.assign_weight()

    def assign_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                module.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.num_features)
        out = self.fc(out)
        return out
