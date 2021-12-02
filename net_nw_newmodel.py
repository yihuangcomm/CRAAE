# Copyright 2018-2020 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================================
# original code is from https://github.com/podgorskiy/GPND/blob/master/net.py
# Modeified by Yi Huang
# =========================================================================================

import torch
from torch import nn
from torch.nn import functional as F


class Generator(nn.Module):
    def __init__(self, z_size, d=32, channels=1):
        super(Generator, self).__init__()
        self.linear0_bn = nn.BatchNorm1d(z_size)
        self.linear1 = nn.Linear(z_size,91*d) 
        self.maxunpool = nn.MaxUnpool2d((1,5))
        self.deconv1_bn = nn.BatchNorm2d(d)     
        self.deconv2 = nn.ConvTranspose2d(d, d, (1,16), 1, 0)
        self.deconv2_bn = nn.BatchNorm2d(d)
        self.deconv3 = nn.ConvTranspose2d(d, d, (1,16), 1, 0) 
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, channels, (1,16), 1, 0) 
        self.d = d

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x, indices):
        x = x.squeeze()
        x = F.relu(self.linear1(self.linear0_bn(x)))
        x = self.maxunpool(x.view(-1,self.d,1,91),indices)
        x = F.relu(self.deconv1_bn(x))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = torch.tanh(self.deconv4(x)) * 0.5 + 0.5
        return x


class Discriminator(nn.Module):
    def __init__(self, d=32, channels=1):
        super(Discriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(channels, d//2, (1,16), 1, 0)
        self.conv2 = nn.Conv2d(d // 2, d*2, (1,16), 1, 0)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, 1, (1,16), 1, 0)
        self.conv3_bn = nn.BatchNorm2d(1)
        self.maxpool = nn.MaxPool2d((1,5),return_indices=True)
        self.flatten = nn.Flatten()
        self.linear4 = nn.Linear(91, 64) #(500-15*3)/5
        self.linear5 = nn.Linear(64,1)
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x,_ = self.maxpool(x)
        x = self.flatten(x)
        x = F.leaky_relu(self.linear4(x), 0.2)
        x = torch.sigmoid(self.linear5(x))
        return x


class Encoder(nn.Module):
    def __init__(self, z_size, d=32, channels=1):
        super(Encoder, self).__init__()
        self.z_size = z_size
        self.conv1 = nn.Conv2d(channels, d, (1,16), 1, 0)
        self.conv1_bn = nn.BatchNorm2d(d)
        self.conv2 = nn.Conv2d(d, d, (1,16), 1, 0)
        self.conv2_bn = nn.BatchNorm2d(d)
        self.conv3 = nn.Conv2d(d, d, (1,16), 1, 0)
        self.conv3_bn = nn.BatchNorm2d(d)
        self.maxpool = nn.MaxPool2d((1,5),return_indices=True)  
        self.flatten = nn.Flatten()      
        self.linear4 = nn.Linear(91*d, z_size) #(500-15*3)/5
        self.linear4_bn = nn.BatchNorm1d(z_size)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.leaky_relu(self.conv1_bn(self.conv1(input)), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x,indices = self.maxpool(x)
        x = self.flatten(x)
        x = self.linear4_bn(self.linear4(x)).view(-1,self.z_size,1,1)

        return x,indices


class ZDiscriminator(nn.Module):
    def __init__(self, z_size, batchSize, d=32):
        super(ZDiscriminator, self).__init__()
        self.linear1 = nn.Linear(z_size, d)
        self.linear2 = nn.Linear(d, d)
        self.linear3 = nn.Linear(d, 1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        x = F.leaky_relu((self.linear1(x)), 0.2)
        x = F.leaky_relu((self.linear2(x)), 0.2)
        x = torch.sigmoid(self.linear3(x))
        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
