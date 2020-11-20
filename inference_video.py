import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_grad_enabled(False)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--video', dest='video', required=True)
parser.add_argument('--montage', dest='montage', action='store_true', help='montage origin video')
parser.add_argument('--skip', dest='skip', action='store_true', help='whether to remove static frames before processing')
parser.add_argument('--fps', dest='fps', type=int, default=None)
parser.add_argument('--png', dest='png', action='store_true', help='whether to output png format outputs')
parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='output video extension')
parser.add_argument('--exp', dest='exp', type=int, default=1)
args = parser.parse_args()
assert (args.exp == 1 or args.exp == 2)
args.exp = 2 ** args.exp

from model.RIFE import Model
model = Model()
model.load_model('./train_log')
model.eval()
model.device()

videoCapture = cv2.VideoCapture(args.video)
fps = np.round(videoCapture.get(cv2.CAP_PROP_FPS))
if args.fps is None:
    args.fps = fps * args.exp
success, frame = videoCapture.read()
h, w, _ = frame.shape
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
if args.png:
    if not os.path.exists('output'):
        os.mkdir('output')
else:
    video_path_wo_ext, ext = os.path.splitext(args.video)
    output = cv2.VideoWriter('{}_{}X_{}fps.{}'.format(video_path_wo_ext, args.exp, int(np.round(args.fps)), args.ext), fourcc, args.fps, (w, h))
    
cnt = 0
def writeframe(frame):
    global cnt
    if args.png:
        cv2.imwrite('output/{:0>7d}.png'.format(cnt), frame)
        cnt += 1
    else:
        output.write(frame)
if args.montage:
    left = w // 4
    w = w // 2
ph = ((h - 1) // 32 + 1) * 32
pw = ((w - 1) // 32 + 1) * 32
padding = (0, pw - w, 0, ph - h)
tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
print('{}.{}, {} frames in total, {}FPS to {}FPS'.format(video_path_wo_ext, args.ext, tot_frame, fps, args.fps))
pbar = tqdm(total=tot_frame)
skip_frame = 1
if args.montage:
    frame = frame[:, left: left + w]
while success:
    lastframe = frame
    success, frame = videoCapture.read()
    if success:
        if args.montage:
            frame = frame[:, left: left + w]
        I0 = torch.from_numpy(np.transpose(lastframe, (2,0,1)).astype("float32") / 255.).to(device).unsqueeze(0)
        I1 = torch.from_numpy(np.transpose(frame, (2,0,1)).astype("float32") / 255.).to(device).unsqueeze(0)
        I0 = F.pad(I0, padding)
        I1 = F.pad(I1, padding)
        p = (F.interpolate(I0, (16, 16), mode='bilinear', align_corners=False)
             - F.interpolate(I1, (16, 16), mode='bilinear', align_corners=False)).abs().mean()
        if p < 5e-3 and args.skip:
            if skip_frame % 100 == 0:
                print("Warning: Your video has {} static frames, skipping them may change the duration of the generated video.".format(skip_frame))
            skip_frame += 1
            pbar.update(1)
            continue
        if p > 0.2:             
            mid1 = lastframe
            mid0 = lastframe
            mid2 = frame
        else:
            mid1 = model.inference(I0, I1)
            if args.exp == 4:
                mid = model.inference(torch.cat((I0, mid1), 0), torch.cat((mid1, I1), 0))
            mid1 = (((mid1[0] * 255.).byte().cpu().detach().numpy().transpose(1, 2, 0)))
            if args.exp == 4:
                mid0 = (((mid[0] * 255.).byte().cpu().detach().numpy().transpose(1, 2, 0)))
                mid2 = (((mid[1] * 255.).byte().cpu().detach().numpy().transpose(1, 2, 0)))
        if args.montage:
            writeframe(np.concatenate((lastframe, lastframe), 1))
            if args.exp == 4:
                writeframe(np.concatenate((lastframe, mid0[:h, :w]), 1))
            writeframe(np.concatenate((lastframe, mid1[:h, :w]), 1))
            if args.exp == 4:
                writeframe(np.concatenate((lastframe, mid2[:h, :w]), 1))
        else:
            writeframe(lastframe)
            if args.exp == 4:
                writeframe(mid0[:h, :w])
            writeframe(mid1[:h, :w])
            if args.exp == 4:
                writeframe(mid2[:h, :w])
        pbar.update(1)
if args.montage:
    writeframe(np.concatenate((lastframe, lastframe), 1))
else:
    writeframe(lastframe)
pbar.close()
output.release()
