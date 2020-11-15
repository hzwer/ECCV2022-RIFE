import os
import cv2
import torch
import argparse
import numpy as np
from torch.nn import functional as F
from model.RIFE import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--video', dest='video', required=True)
parser.add_argument('--montage', dest='montage', action='store_true', help='montage origin video')
args = parser.parse_args()

model = Model()
model.load_model('./train_log')
model.eval()
model.device()

videoCapture = cv2.VideoCapture(args.video)
fps = videoCapture.get(cv2.CAP_PROP_FPS)
success, frame = videoCapture.read()
h, w, _ = frame.shape
ph = ((h - 1) // 32 + 1) * 32
pw = ((w - 1) // 32 + 1) * 32
padding = (0, pw - w, 0, ph - h)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
print('{}.mp4'.format(args.video[:-4]))
if args.montage:
    output = cv2.VideoWriter('{}_2x.mp4'.format(args.video[:-4]), fourcc, fps * 2, (2*w, h))
else:        
    output = cv2.VideoWriter('{}_2x.mp4'.format(args.video[:-4]), fourcc, fps * 2, (w, h))
frame = frame
while success:
    lastframe = frame
    success, frame = videoCapture.read()
    if success:
        I0 = torch.from_numpy(np.transpose(lastframe, (2,0,1)).astype("float32") / 255.).to(device).unsqueeze(0)
        I1 = torch.from_numpy(np.transpose(frame, (2,0,1)).astype("float32") / 255.).to(device).unsqueeze(0)
        I0 = F.pad(I0, padding)
        I1 = F.pad(I1, padding)
        if (F.interpolate(I0, (16, 16), mode='bilinear', align_corners=False)
            - F.interpolate(I1, (16, 16), mode='bilinear', align_corners=False)).abs().mean() > 0.2:            
            mid = lastframe
        else:
            mid = model.inference(I0, I1)
            mid = ((mid[0].cpu().detach().numpy().transpose(1, 2, 0))*255.).astype('uint8')
        if args.montage:
            output.write(torch.cat((lastframe, lastframe), 2))
            output.write(torch.cat((lastframe, mid[:h, :w]), 2))
        else:
            output.write(lastframe)
            output.write(mid[:h, :w])
    if args.montage:
        output.write(torch.cat((frame, frame), 2))
    else:
        output.write(frame)
output.release()
