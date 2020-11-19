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
parser.add_argument('--skip', dest='skip', action='store_true', help='whether to remove static frames before processing')
parser.add_argument('--fps', dest='fps', type=int, default=None)
parser.add_argument('--png', dest='png', action='store_true', help='whether to output png format outputs')
parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='output video extension')
parser.add_argument('--times', dest='times', type=int, default=1)
args = parser.parse_args()
assert (args.times == 1 or args.times == 2)
args.times = 2 ** args.times

from model.RIFE import Model
model = Model()
model.load_model('./train_log')
model.eval()
model.device()

videoCapture = cv2.VideoCapture(args.video)
fps = np.round(videoCapture.get(cv2.CAP_PROP_FPS))
success, frame = videoCapture.read()
h, w, _ = frame.shape
if args.fps is None:
    args.fps = fps * args.times
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
if args.png:
    if not os.path.exists('output'):
        os.mkdir('output')
else:
    video_path_wo_ext, ext = os.path.splitext(args.video)
    output = cv2.VideoWriter('{}_{}X_{}fps.{}'.format(video_path_wo_ext, args.times, int(np.round(args.fps)), args.ext), fourcc, args.fps, (w, h))
    
cnt = 0
skip_frame = 1
def writeframe(I0, mid0, mid1, mid2, I1, p):
    global cnt, skip_frame, args
    for i in range(I0.shape[0]):
        if p[i] > 0.2:
            if args.times == 4:
                mid0[i] = I0[i]
            mid1[i] = I0[i]
            if args.times == 4:
                mid2[i] = I1[i]
        if p[i] < 2e-3 and args.skip:
            if skip_frame % 100 == 0:
                print("Warning: Your video has {} static frames, skipping them may change the duration of the generated video.".format(skip_frame))
            skip_frame += 1
            continue
        if args.png:
            cv2.imwrite('output/{:0>7d}.png'.format(cnt), I0[i])
            cnt += 1
            if args.times == 4:
                cv2.imwrite('output/{:0>7d}.png'.format(cnt), mid0[i])
                cnt += 1
            cv2.imwrite('output/{:0>7d}.png'.format(cnt), mid1[i])
            cnt += 1
            if args.times == 4:
                cv2.imwrite('output/{:0>7d}.png'.format(cnt), mid2[i])
                cnt += 1
        else:
            output.write(I0[i])
            if args.times == 4:
                output.write(mid0[i])
            output.write(mid1[i])
            if args.times == 4:
                output.write(mid2[i])


ph = ((h - 1) // 32 + 1) * 32
pw = ((w - 1) // 32 + 1) * 32
padding = (0, pw - w, 0, ph - h)
tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
print('{}.{}, {} frames in total, {}FPS to {}FPS'.format(video_path_wo_ext, args.ext, tot_frame, fps, args.fps))
pbar = tqdm(total=tot_frame)
img_list = [frame]
while success:
    success, frame = videoCapture.read()
    if success:
        img_list.append(frame)
    if len(img_list) == 5 or (not success and len(img_list) > 1):
        I0 = torch.from_numpy(np.transpose(img_list[:-1], (0, 3, 1, 2)).astype("float32") / 255.).to(device)
        I1 = torch.from_numpy(np.transpose(img_list[1:], (0, 3, 1, 2)).astype("float32") / 255.).to(device)
        p = (F.interpolate(I0, (16, 16), mode='bilinear', align_corners=False)
             - F.interpolate(I1, (16, 16), mode='bilinear', align_corners=False)).abs()
        I0 = F.pad(I0, padding)
        I1 = F.pad(I1, padding)
        mid1 = model.inference(I0, I1)
        if args.times == 4:
            mid0 = model.inference(I0, mid1)
            mid2 = model.inference(mid1, I1)
        I0 = ((I0[:, :, :h, :w] * 255.).cpu().detach().numpy().transpose(0, 2, 3, 1)).astype('uint8')
        I1 = ((I1[:, :, :h, :w] * 255.).cpu().detach().numpy().transpose(0, 2, 3, 1)).astype('uint8')
        mid1 = ((mid1[:, :, :h, :w] * 255.).cpu().detach().numpy().transpose(0, 2, 3, 1)).astype('uint8')
        if args.times == 4:
            mid0 = ((mid0[:, :, :h, :w] * 255.).cpu().detach().numpy().transpose(0, 2, 3, 1)).astype('uint8')
            mid2 = ((mid2[:, :, :h, :w] * 255.).cpu().detach().numpy().transpose(0, 2, 3, 1)).astype('uint8')
        else:
            mid0, mid2 = None, None
        writeframe(I0, mid0, mid1, mid2, I1, p.mean(3).mean(2).mean(1))
        pbar.update(4)
        img_list = img_list[-1:]
pbar.close()
