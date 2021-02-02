import cv2
import sys
sys.path.append('.')
import time
import torch
import torch.nn as nn
from model.RIFE import Model

model = Model()
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
       
I0 = torch.rand(1, 3, 480, 640).to(device)
I1 = torch.rand(1, 3, 480, 640).to(device)
with torch.no_grad():
    for i in range(100):
        pred = model.inference(I0, I1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time_stamp = time.time()
    for i in range(100):
        pred = model.inference(I0, I1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print((time.time() - time_stamp) / 100)
