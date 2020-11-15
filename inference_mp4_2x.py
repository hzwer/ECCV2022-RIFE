import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_grad_enabled(False)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--video', dest='video', required=True)
parser.add_argument('--montage', dest='montage', action='store_true', help='montage origin video')
parser.add_argument('--fps', dest='fps', type=int, default=60)
parser.add_argument('--model', dest='model', type=str, default='RIFE')
args = parser.parse_args()

if args.model == '2F':
    from model.RIFE2F import Model
else:
    from model.RIFE import Model
model = Model()
model.load_model('./train_log')
model.eval()
model.device()

videoCapture = cv2.VideoCapture(args.video)
fps = np.round(videoCapture.get(cv2.CAP_PROP_FPS))
success, frame = videoCapture.read()
h, w, _ = frame.shape
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
output = cv2.VideoWriter('{}_4x.mp4'.format(args.video[:-4]), fourcc, args.fps, (w, h))
if args.montage:
    left = w // 4
    w = w // 2
ph = ((h - 1) // 32 + 1) * 32
pw = ((w - 1) // 32 + 1) * 32
padding = (0, pw - w, 0, ph - h)
tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
print('{}.mp4, {} frames in total, {}FPS to {}FPS'.format(args.video[:-4], tot_frame, fps, args.fps))
pbar = tqdm(total=tot_frame)
if args.montage:
    frame = frame[:, left: left + w]
while success:
    lastframe = frame
    success, frame = videoCapture.read()
    if success:
        if args.montage:
            frame = frame[:, left: left + w]if args.montage:            
        I0 = torch.from_numpy(np.transpose(lastframe, (2,0,1)).astype("float32") / 255.).to(device).unsqueeze(0)
        I1 = torch.from_numpy(np.transpose(frame, (2,0,1)).astype("float32") / 255.).to(device).unsqueeze(0)
        I0 = F.pad(I0, padding)
        I1 = F.pad(I1, padding)
        if (F.interpolate(I0, (16, 16), mode='bilinear', align_corners=False)
            - F.interpolate(I1, (16, 16), mode='bilinear', align_corners=False)).abs().mean() > 0.2:
            mid1 = lastframe
        else:
            mid1 = model.inference(I0, I1)
            mid1 = (((mid1[0] * 255.).cpu().detach().numpy().transpose(1, 2, 0))).astype('uint8')
        if args.montage:
            output.write(np.concatenate((lastframe, lastframe), 1))
            output.write(np.concatenate((lastframe, mid1[:h, :w]), 1))
        else:
            output.write(lastframe)
            output.write(mid1[:h, :w])
        pbar.update(1)
if args.montage:
    output.write(np.concatenate((lastframe, lastframe), 1))
else:
    output.write(lastframe)
pbar.close()
output.release()
