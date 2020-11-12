import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EPE(nn.Module):
    def __init__(self):
        super(EPE, self).__init__()

    def forward(self, flow, gt, loss_mask):
        loss_map = (flow - gt.detach()) ** 2
        loss_map = (loss_map.sum(1, True) + 1e-6) ** 0.5
        return (loss_map * loss_mask)


class Ternary(nn.Module):
    def __init__(self):
        super(Ternary, self).__init__()
        patch_size = 7
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape(
            (patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w).float().to(device)

    def transform(self, img):
        patches = F.conv2d(img, self.w, padding=3, bias=None)
        transf = patches - img
        transf_norm = transf / torch.sqrt(0.81 + transf**2)
        return transf_norm

    def rgb2gray(self, rgb):
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def hamming(self, t1, t2):
        dist = (t1 - t2) ** 2
        dist_norm = torch.mean(dist / (0.1 + dist), 1, True)
        return dist_norm

    def valid_mask(self, t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        mask = F.pad(inner, [padding] * 4)
        return mask

    def forward(self, img0, img1):
        img0 = self.transform(self.rgb2gray(img0))
        img1 = self.transform(self.rgb2gray(img1))
        return self.hamming(img0, img1) * self.valid_mask(img0, 1)


class SOBEL(nn.Module):
    def __init__(self):
        super(SOBEL, self).__init__()
        self.kernelX = torch.tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1],
        ]).float()
        self.kernelY = self.kernelX.clone().T
        self.kernelX = self.kernelX.unsqueeze(0).unsqueeze(0).to(device)
        self.kernelY = self.kernelY.unsqueeze(0).unsqueeze(0).to(device)

    def forward(self, pred, gt):
        N, C, H, W = pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3]
        img_stack = torch.cat(
            [pred.reshape(N*C, 1, H, W), gt.reshape(N*C, 1, H, W)], 0)
        sobel_stack_x = F.conv2d(img_stack, self.kernelX, padding=1)
        sobel_stack_y = F.conv2d(img_stack, self.kernelY, padding=1)
        pred_X, gt_X = sobel_stack_x[:N*C], sobel_stack_x[N*C:]
        pred_Y, gt_Y = sobel_stack_y[:N*C], sobel_stack_y[N*C:]

        L1X, L1Y = torch.abs(pred_X-gt_X), torch.abs(pred_Y-gt_Y)
        loss = (L1X+L1Y)
        return loss


if __name__ == '__main__':
    img0 = torch.zeros(3, 3, 256, 256).float().to(device)
    img1 = torch.tensor(np.random.normal(
        0, 1, (3, 3, 256, 256))).float().to(device)
    ternary_loss = Ternary()
    print(ternary_loss(img0, img1).shape)
