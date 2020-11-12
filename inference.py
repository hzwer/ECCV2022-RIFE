import cv2
import torch
from model.RIFE import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model()
model.load_model('./train_log')
model.eval()
model.device()

img0 = cv2.imread('0.png')
img1 = cv2.imread('1.png')

h, w, _ = img0.shape

img0 = torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.
img1 = torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.

imgs = torch.cat((img0, img1), 0).float()

with torch.no_grad():
    res = model.inference(imgs.unsqueeze(0)) * 255
cv2.imwrite('out.png', res[0].numpy().transpose(1, 2, 0))
