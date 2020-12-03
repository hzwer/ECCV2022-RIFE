import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import skvideo.io
from queue import Queue, Empty
import moviepy.editor
import shutil
warnings.filterwarnings("ignore")

def transferAudio(sourceVideo, targetVideo):
    tempAudioFileName = "./temp/audio.mp3"
    try:
        # split audio from original video file and store in "temp" directory
        if True:
            # extract audio from video
            if True:
                video = moviepy.editor.VideoFileClip(sourceVideo)
                audio = video.audio
            # clear old "temp" directory if it exits
            if os.path.isdir("temp"):
                # remove temp directory
                shutil.rmtree("temp")
            # create new "temp" directory
            os.makedirs("temp")
            # write audio file to "temp" directory
            audio.write_audiofile(tempAudioFileName)
            os.rename(targetVideo, "noAudio_"+targetVideo)
        # combine audio file and new video file
        os.system("ffmpeg -y -i " + "noAudio_"+targetVideo + " -i " + tempAudioFileName + " -c copy " + targetVideo)
        # remove audio-less video
        os.remove("noAudio_"+targetVideo)
    except:
        pass
    # remove temp directory
    shutil.rmtree("temp")

    
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
parser.add_argument('--png', dest='png', action='store_true', help='whether to vid_out png format vid_outs')
parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='vid_out video extension')
parser.add_argument('--exp', dest='exp', type=int, default=1)
args = parser.parse_args()
assert (args.exp == 1 or args.exp == 2)
args.exp = 2 ** args.exp

from model.RIFE_HD import Model
model = Model()
model.load_model('./train_log', -1)
model.eval()
model.device()

videoCapture = cv2.VideoCapture(args.video)
fps = videoCapture.get(cv2.CAP_PROP_FPS)
tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
videoCapture.release()
if args.fps is None:
    fpsNotAssigned = True
    args.fps = fps * args.exp
else:
    fpsNotAssigned = False
videogen = skvideo.io.vreader(args.video)
lastframe = next(videogen)
h, w, _ = lastframe.shape
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
if args.png:
    if not os.path.exists('vid_out'):
        os.mkdir('vid_out')
else:
    video_path_wo_ext, ext = os.path.splitext(args.video)
    vid_out = cv2.VideoWriter('{}_{}X_{}fps.{}'.format(video_path_wo_ext, args.exp, int(np.round(args.fps)), args.ext), fourcc, args.fps, (w, h))
    
def clear_buffer(user_args, buffer):
    cnt = 0
    while True:
        item = buffer.get()
        if item is None:
            break
        if user_args.png:
            cv2.imwrite('vid_out/{:0>7d}.png'.format(cnt), item[:, :, ::-1])
            cnt += 1
        else:
            vid_out.write(item[:, :, ::-1])
            
if args.montage:
    left = w // 4
    w = w // 2
ph = ((h - 1) // 32 + 1) * 32
pw = ((w - 1) // 32 + 1) * 32
padding = (0, pw - w, 0, ph - h)
print('{}.{}, {} frames in total, {}FPS to {}FPS'.format(video_path_wo_ext, args.ext, tot_frame, fps, args.fps))
pbar = tqdm(total=tot_frame)
skip_frame = 1
if args.montage:
    lastframe = lastframe[:, left: left + w]
buffer = Queue()
_thread.start_new_thread(clear_buffer, (args, buffer))
for frame in videogen:
    if args.montage:
        frame = frame[:, left: left + w]
    I0 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
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
        mid2 = lastframe
    else:
        if args.exp == 4:
            mid1 = model.inference(I0, I1)
            mid0 = model.inference(I0, mid1)
            mid2 = model.inference(mid1, I1)
            mid0 = (((mid0[0] * 255.).byte().cpu().detach().numpy().transpose(1, 2, 0)))
            mid1 = (((mid1[0] * 255.).byte().cpu().detach().numpy().transpose(1, 2, 0)))
            mid2 = (((mid2[0] * 255.).byte().cpu().detach().numpy().transpose(1, 2, 0)))
        else:
            mid1 = model.inference(I0, I1)
            mid1 = (((mid1[0] * 255.).byte().cpu().detach().numpy().transpose(1, 2, 0)))
    if args.montage:
        buffer.put(np.concatenate((lastframe, lastframe), 1))
        if args.exp == 4:            
            buffer.put(np.concatenate((lastframe, mid0[:h, :w]), 1))
            buffer.put(np.concatenate((lastframe, mid1[:h, :w]), 1))
            buffer.put(np.concatenate((lastframe, mid2[:h, :w]), 1))
        else:            
            buffer.put(np.concatenate((lastframe, mid1[:h, :w]), 1))
    else:
        buffer.put(lastframe)
        if args.exp == 4:
            buffer.put(mid0[:h, :w])
            buffer.put(mid1[:h, :w])
            buffer.put(mid2[:h, :w])
        else:
            buffer.put(mid1[:h, :w])
    pbar.update(1)
    lastframe = frame
if args.montage:
    buffer.put(np.concatenate((lastframe, lastframe), 1))
else:
    buffer.put(lastframe)
import time
while(not buffer.empty()):
    time.sleep(0.1)
pbar.close()
if not vid_out is None:
    vid_out.release()

# move audio to new video file if appropriate
if args.png == False and fpsNotAssigned == True and not args.skip:
    outputVideoFileName = '{}_{}X_{}fps.{}'.format(video_path_wo_ext, args.exp, int(np.round(args.fps)), args.ext)
    transferAudio(video_path_wo_ext + "." + args.ext, outputVideoFileName)
