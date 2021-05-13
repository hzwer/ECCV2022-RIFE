import torch
import torch.nn as nn
import torch.nn.functional as F
from model.warplayer import warp

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        nn.PReLU(out_planes)
    )

def conv_wo_act(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        )

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c, 3, 2, 1),
            conv(c, 2*c, 3, 2, 1),
            )
        self.convblock0 = nn.Sequential(
            conv(2*c, 2*c),
            conv(2*c, 2*c),
        )
        self.convblock1 = nn.Sequential(
            conv(2*c, 2*c),
            conv(2*c, 2*c),
        )
        self.convblock2 = nn.Sequential(
            conv(2*c, 2*c),
            conv(2*c, 2*c),
        )
        self.conv1 = nn.ConvTranspose2d(2*c, 4, 4, 2, 1)

    def forward(self, x, flow=None, scale=1):
        x = F.interpolate(x, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
        if flow != None:
            flow = F.interpolate(flow, scale_factor= 1. / scale, mode="bilinear", align_corners=False) * (1. / scale)
            x = torch.cat((x, flow), 1)
        x = self.conv0(x)
        x = self.convblock0(x) + x
        x = self.convblock1(x) + x
        x = self.convblock2(x) + x
        x = self.conv1(x)
        flow = x
        if scale != 1:
            flow = F.interpolate(flow, scale_factor= scale, mode="bilinear", align_corners=False) * scale
        return flow
    
class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(6, c=80)
        self.block1 = IFBlock(10, c=80)
        self.block2 = IFBlock(10, c=80)

    def forward(self, x, scale_list=[4,2,1]):
        flow0 = self.block0(x, scale=scale_list[0])
        F1 = flow0
        F1_large = F.interpolate(F1, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        warped_img0 = warp(x[:, :3], F1_large[:, :2])
        warped_img1 = warp(x[:, 3:], F1_large[:, 2:4])
        flow1 = self.block1(torch.cat((warped_img0, warped_img1), 1), F1_large, scale=scale_list[1])
        F2 = (flow0 + flow1)
        F2_large = F.interpolate(F2, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        warped_img0 = warp(x[:, :3], F2_large[:, :2])
        warped_img1 = warp(x[:, 3:], F2_large[:, 2:4])
        flow2 = self.block2(torch.cat((warped_img0, warped_img1), 1), F2_large, scale=scale_list[2])
        F3 = (flow0 + flow1 + flow2)
        return F3, [F1, F2, F3]
