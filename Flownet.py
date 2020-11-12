import torch
import torch.nn as nn
import torch.nn.functional as F
from warplayer import warp

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_planes),
        nn.PReLU(out_planes)
    )

def conv_wo_act(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes),
        )

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.PReLU(out_planes)
    )

class ResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(ResBlock, self).__init__()
        if in_planes == out_planes and stride == 1:
            self.conv0 = nn.Identity()
        else:
            self.conv0 = nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv_wo_act(out_planes, out_planes, 3, 1, 1)
        self.relu1 = nn.PReLU(1)
        self.relu2 = nn.PReLU(out_planes)
        self.fc1 = nn.Conv2d(out_planes, 16, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(16, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        y = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        w = x.mean(3, True).mean(2, True)
        w = self.relu1(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        x = self.relu2(x * w + y)
        return x

class Flownet(nn.Module):
    def __init__(self, in_planes, scale=1, c=64):
        super(Flownet, self).__init__()
        self.scale = scale
        self.conv0 = conv(in_planes, c, 3, 2, 1)
        self.res0 = ResBlock(c, c)
        self.res1 = ResBlock(c, c)
        self.res2 = ResBlock(c, c)
        self.res3 = ResBlock(c, c)
        self.res4 = ResBlock(c, c)
        self.res5 = ResBlock(c, c)
        self.conv1 = nn.Conv2d(c, 8, 3, 1, 1)
        self.up = nn.PixelShuffle(2)

    def forward(self, x):
        if self.scale != 1:
            x = F.interpolate(x, scale_factor= 1. / self.scale, mode="bilinear", align_corners=False, recompute_scale_factor=False)
        x = self.conv0(x)
        x = self.res0(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.conv1(x)
        flow = self.up(x)
        if self.scale != 1:
            flow = F.interpolate(flow, scale_factor= self.scale, mode="bilinear", align_corners=False, recompute_scale_factor=False)
        return flow
    
class FlownetCas(nn.Module):
    def __init__(self):
        super(FlownetCas, self).__init__()
        self.block0 = Flownet(6, scale=4, c=192)
        self.block1 = Flownet(8, scale=2, c=128)
        self.block2 = Flownet(8, scale=1, c=64)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False)
        flow0 = self.block0(x)
        F1 = flow0
        warped_img0 = warp(x[:, :3], F1)
        warped_img1 = warp(x[:, 3:], -F1)
        flow1 = self.block1(torch.cat((warped_img0, warped_img1, F1), 1))
        F2 = (flow0 + flow1)
        warped_img0 = warp(x[:, :3], F2)
        warped_img1 = warp(x[:, 3:], -F2)
        flow2 = self.block2(torch.cat((warped_img0, warped_img1, F2), 1))
        F3 = (flow0 + flow1 + flow2)
        return F3, [F1, F2, F3]
