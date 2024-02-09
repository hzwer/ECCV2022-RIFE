import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def warp(tenInput, tenFlow):
    device = tenInput.device
    tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
        1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
    tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
        1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
    backwarp_tenGrid = torch.cat(
        [tenHorizontal, tenVertical], 1).to(device)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),        
        nn.LeakyReLU(0.2, True)
    )

def conv_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, True)
    )
    
class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.cnn0 = nn.Conv2d(3, 32, 3, 2, 1)
        self.cnn1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.cnn2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(32, 8, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x0 = self.cnn0(x)
        x = self.relu(x0)
        x1 = self.cnn1(x)
        x = self.relu(x1)
        x2 = self.cnn2(x)
        x = self.relu(x2)
        x3 = self.cnn3(x)
        return x3

class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super(ResConv, self).__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1\
)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock = nn.Sequential(
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
        )
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4*6, 4, 2, 1),
            nn.PixelShuffle(2)
        )

    def forward(self, x, flow: Optional[torch.Tensor]=None, scale: float=1.0):
        x = F.interpolate(x, scale_factor= 1. / scale, mode="bilinear", align_corners=False, recompute_scale_factor=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor= 1. / scale, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=False)
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        return flow, mask
        
class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7+16, c=192)
        self.block1 = IFBlock(8+4+16, c=128)
        self.block2 = IFBlock(8+4+16, c=96)
        self.block3 = IFBlock(8+4+16, c=64)
        self.encode = Head()        

    def forward(self, x, timestep: float=0.5, scale_list: List[float]=(8.0, 4.0, 2.0, 1.0)):
        channel = x.shape[1] // 2
        img0 = x[:, :channel]
        img1 = x[:, channel:]
        timestep = (x[:, :1].clone() * 0 + 1) * timestep
        f0 = self.encode(img0[:, :3])
        f1 = self.encode(img1[:, :3])
        warped_img0 = img0
        warped_img1 = img1
        flow: Optional[torch.Tensor] = None
        mask: Optional[torch.Tensor] = None

        # self.block0
        flow, mask = self.block0(torch.cat((img0[:, :3], img1[:, :3], f0, f1, timestep), 1), None, scale=scale_list[0])
        warped_img0 = warp(img0, flow[:, :2])
        warped_img1 = warp(img1, flow[:, 2:4])

        # self.block1
        wf0 = warp(f0, flow[:, :2])
        wf1 = warp(f1, flow[:, 2:4])
        fd, m0 = self.block1(torch.cat((warped_img0[:, :3], warped_img1[:, :3], wf0, wf1, timestep, mask), 1), flow, scale=scale_list[1])
        mask = m0
        flow = flow + fd
        warped_img0 = warp(img0, flow[:, :2])
        warped_img1 = warp(img1, flow[:, 2:4])
        
        # self.block2
        wf0 = warp(f0, flow[:, :2])
        wf1 = warp(f1, flow[:, 2:4])
        fd, m0 = self.block2(torch.cat((warped_img0[:, :3], warped_img1[:, :3], wf0, wf1, timestep, mask), 1), flow, scale=scale_list[2])
        mask = m0
        flow = flow + fd
        warped_img0 = warp(img0, flow[:, :2])
        warped_img1 = warp(img1, flow[:, 2:4])

        # self.block3
        wf0 = warp(f0, flow[:, :2])
        wf1 = warp(f1, flow[:, 2:4])
        fd, m0 = self.block3(torch.cat((warped_img0[:, :3], warped_img1[:, :3], wf0, wf1, timestep, mask), 1), flow, scale=scale_list[3])
        mask = m0
        flow = flow + fd
        warped_img0 = warp(img0, flow[:, :2])
        warped_img1 = warp(img1, flow[:, 2:4])

        mask = torch.sigmoid(mask)
        warped_frame = warped_img0 * mask + warped_img1 * (1 - mask)
        return flow, mask, warped_frame
