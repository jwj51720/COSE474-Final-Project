import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class BasicBlock(nn.Module):
    mul = 1

    def __init__(self, in_channel, out_channel, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channel)
        self.shortcut = nn.Sequential()

        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channel, out_channel, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channel),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn(out)
        out += self.shortcut(x)  # Skip Connection
        out = F.relu(out)
        return out


class BottleNeck(nn.Module):
    mul = 4

    def __init__(self, in_channel, out_channel, stride=1):
        super(BottleNeck, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channel, out_channel, kernel_size=1, stride=stride, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.conv3 = nn.Conv2d(
            out_channel, out_channel * self.mul, kernel_size=1, stride=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel * self.mul)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channel != out_channel * self.mul:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channel,
                    out_channel * self.mul,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channel * self.mul),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(self.in_channel)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(512 * block.mul, num_classes)

    def make_layer(self, block, out_channel, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            layers.append(block(self.in_channel, out_channel, strides[i]))
            self.in_channel = block.mul * out_channel
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


class NewModel(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.method = config["method"]
        self.model1 = copy.deepcopy(model)
        self.model2 = copy.deepcopy(model)
        self.num_classes_per_model = len(config["classes1"])
        self.weight1 = nn.Parameter(torch.randn((1, 1)))
        self.weight2 = nn.Parameter(torch.randn((1, 1)))
        self.linear1 = nn.Linear(1, 5)
        self.linear2 = nn.Linear(1, 5)
        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.model1(x)
        model1_others = x1[:, -1].unsqueeze(1)
        x1 = x1[:, : self.num_classes_per_model]

        x2 = self.model2(x)
        model2_others = x2[:, -1].unsqueeze(1)
        x2 = x2[:, : self.num_classes_per_model]

        if self.method == "ml":  # mul / linear&activation
            x1 = x1 * self.relu(self.linear1(model2_others))
            x2 = x2 * self.relu(self.linear2(model1_others))
        elif self.method == "mp":  # mul / parameter
            x1 = x1 * model2_others * self.weight2
            x2 = x2 * model1_others * self.weight1
        elif self.method == "pl":  # plus / linear&activation
            x1 = x1 + self.relu(self.linear1(model2_others))
            x2 = x2 + self.relu(self.linear2(model1_others))
        elif self.method == "pp":  # plus / parameter
            x1 = x1 + model2_others * self.weight2
            x2 = x2 + model1_others * self.weight1

        output = torch.cat((x1, x2), dim=1)
        return output
