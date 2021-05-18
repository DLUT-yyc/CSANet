import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import os
import sys
import pdb
import numpy as np
from torch.autograd import Variable
import functools
affine_par = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, './utils'))
from resnet_block import conv3x3, Bottleneck
class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1, num_classes=19):
        super(SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_classes = num_classes
        if out_channels == None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            nn.SyncBatchNorm(self.key_channels),
        )

        self.f_query_x = self.f_key
        self.f_query_y = self.f_key
        
        self.f_value_x = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
            kernel_size=1, stride=1, padding=0)
        self.f_value_y = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
            kernel_size=1, stride=1, padding=0)

        self.conv51 = nn.Sequential(nn.Conv2d(self.value_channels, self.value_channels, 3, padding=1, bias=False),
                                   nn.SyncBatchNorm(self.value_channels),
                                   nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(self.value_channels, self.value_channels, 3, padding=1, bias=False),
                                   nn.SyncBatchNorm(self.value_channels),
                                   nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(self.value_channels, self.out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(self.value_channels, self.out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(self.value_channels, self.out_channels, 1))
        
        self.gamma2 = nn.Parameter(torch.zeros(1))
        

        self.cls1 = nn.Sequential(
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
        self.cls2 = nn.Sequential(
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x, y):    
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)
        ############################## value x y ###############################
        value_x = self.f_value_x(x)
        value_x_resize = value_x.view(batch_size, self.value_channels, -1).permute(0, 2, 1)
        value_y = self.f_value_y(y)
        value_y_resize = value_y.view(batch_size, self.value_channels, -1).permute(0, 2, 1)

        ################################ key x ############################
        key_x = self.f_key(x).view(batch_size, self.key_channels, -1)

        ############################# query x y ##########################
        query_x = self.f_query_x(x).view(batch_size, self.key_channels, -1).permute(0, 2, 1)
        query_y = self.f_query_y(y).view(batch_size, self.key_channels, -1).permute(0, 2, 1)

        ########################### sim_map ########################### 
        sim_map_xy = torch.matmul(query_y, key_x)    # (batch_size, HW, HW)
        sim_map_xx = torch.matmul(query_x, key_x)   
      
        sim_map_xy = (self.key_channels**-.5) * sim_map_xy
        sim_map_xx = (self.key_channels**-.5) * sim_map_xx

        sim_map_xy = F.softmax(sim_map_xy, dim=-1)
        sim_map_xx = F.softmax(sim_map_xx, dim=-1)
        
        ########################## warp ####################
        x_cls = self.cls1(value_x)
        y_cls = self.cls2(value_y)
        # x_cls_resize = x_cls.view(batch_size, self.num_classes, -1).permute(0, 2, 1)
        y_cls_resize = y_cls.view(batch_size, self.num_classes, -1).permute(0, 2, 1)

        warp_yx = torch.matmul(sim_map_xy, y_cls_resize)
        warp_yx = warp_yx.permute(0, 2, 1).contiguous()
        warp_yx = warp_yx.view(batch_size, self.num_classes, *x.size()[2:])

        ########################## context ########################### 
        context_xx = torch.matmul(sim_map_xx, value_x_resize)
        context_xy = torch.matmul(sim_map_xy, value_y_resize)

        context_xx = context_xx.permute(0, 2, 1).contiguous()
        context_xy = context_xy.permute(0, 2, 1).contiguous()

        context_xx = context_xx.view(batch_size, self.value_channels, *x.size()[2:])
        context_xy = context_xy.view(batch_size, self.value_channels, *x.size()[2:])

        context_xx = context_xx + x
        # context_xy = context_xy

        ######################## context fusion #########################
        context_xx = self.conv51(context_xx)
        context_xy = self.conv52(context_xy)

        context_xx = self.conv6(context_xx)
        context_xy = self.conv7(context_xy)

        context = context_xx + context_xy*self.gamma2
        context = self.conv8(context)

        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)
        return context, x_cls, warp_yx, y_cls

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.SyncBatchNorm(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.SyncBatchNorm(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.SyncBatchNorm(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1,1,1)) # we do not apply multi-grid method here
        
        self.conv_context = nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1)
        self.syncbn = nn.SyncBatchNorm(512)
        # extra added layers
        self.context = SelfAttentionBlock(in_channels=512, out_channels=512, key_channels=256, value_channels=512, num_classes=19)
        
        self.cls = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        self.dsn = nn.Sequential(
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
        self.conv_final = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1, bias=False),
                                   nn.SyncBatchNorm(512),
                                   nn.ReLU())

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.SyncBatchNorm(planes * block.expansion,affine = affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))
        return nn.Sequential(*layers)

    def forward(self, x, y):    # 3 add y
        x = self.relu1(self.bn1(self.conv1(x))); y = self.relu1(self.bn1(self.conv1(y)))
        x = self.relu2(self.bn2(self.conv2(x))); y = self.relu2(self.bn2(self.conv2(y)))
        x = self.relu3(self.bn3(self.conv3(x))); y = self.relu3(self.bn3(self.conv3(y)))
        x = self.maxpool(x); y = self.maxpool(y)
        x = self.layer1(x);  y = self.layer1(y)
        x = self.layer2(x);  y = self.layer2(y)
        x = self.layer3(x);  y = self.layer3(y)

        x = self.layer4(x);  y = self.layer4(y)
        x = self.conv_context(x);   y = self.conv_context(y)
        x = self.syncbn(x); y = self.syncbn(y)
        x = self.relu(x); y = self.relu(y)
        x_dsn = self.dsn(x)

        x, x_cls, warp_yx, y_cls= self.context(x, y)
        x = self.conv_final(x)
        x = self.cls(x)

        return [x, x_dsn, x_cls, warp_yx, y_cls]


def get_resnet101_base_oc_dsn(num_classes=21):
    model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes)
    return model

if __name__ == "__main__":
    model = get_resnet101_base_oc_dsn(19).cuda()
    x1 = torch.randn(1, 3, 769, 769).cuda()
    x2 = torch.randn(1, 3, 769, 769).cuda()
    y = model(x1, x2, 1)

