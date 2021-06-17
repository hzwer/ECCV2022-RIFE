import os
import sys
sys.path.append('.')
import cv2
import math
import torch
import argparse
import numpy as np
from torch.nn import functional as F
from model.pytorch_msssim import ssim_matlab
from model.RIFE import Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model()
model.load_model('train_log')
model.eval()
model.device()

name = ['Beanbags', 'Dimetrodon', 'DogDance', 'Grove2', 'Grove3', 'Hydrangea', 'MiniCooper', 'RubberWhale', 'Urban2', 'Urban3', 'Venus', 'Walking']
IE_list = []
for i in name:
    i0 = cv2.imread('other-data/{}/frame10.png'.format(i)).transpose(2, 0, 1) / 255.
    i1 = cv2.imread('other-data/{}/frame11.png'.format(i)).transpose(2, 0, 1) / 255.
    gt = cv2.imread('other-gt-interp/{}/frame10i11.png'.format(i)) 
    h, w = i0.shape[1], i0.shape[2]
    imgs = torch.zeros([1, 6, 480, 640]).to(device)
    ph = (480 - h) // 2
    pw = (640 - w) // 2
    imgs[:, :3, :h, :w] = torch.from_numpy(i0).unsqueeze(0).float().to(device)
    imgs[:, 3:, :h, :w] = torch.from_numpy(i1).unsqueeze(0).float().to(device)
    I0 = imgs[:, :3]
    I2 = imgs[:, 3:]
    pred = model.inference(I0, I2)
    out = pred[0].detach().cpu().numpy().transpose(1, 2, 0)
    out = np.round(out[:h, :w] * 255)
    IE_list.append(np.abs((out - gt * 1.0)).mean())
    print(np.mean(IE_list))
