import cv2
import ast
import torch
import ujson as json
import nori2 as nori
import numpy as np
import random
# import imgaug.augmenters as iaa
from torch.utils.data import DataLoader, Dataset

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class VimeoDataset(Dataset):
    def __init__(self, dataset_name, batch_size=32):
        self.batch_size = batch_size
        self.path = 'XXX'
        self.dataset_name = dataset_name
        self.load_data()
        self.nf = nori.Fetcher()
        self.h = 256
        self.w = 448
        xx = np.arange(0, self.w).reshape(1,-1).repeat(self.h,0)
        yy = np.arange(0, self.h).reshape(-1,1).repeat(self.w,1)
        self.grid = np.stack((xx,yy),2).copy()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        self.train_data = XXX
        self.flow_data = XXX
        self.val_data = XXX
        if self.dataset_name == 'train':
            self.meta_data = self.train_data
        else:
            self.meta_data = self.val_data
        self.nr_sample = len(self.meta_data)        

    def aug(self, img0, gt, img1, flow_gt, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        flow_gt = flow_gt[x:x+h, y:y+w, :]
        return img0, gt, img1, flow_gt

    def getimg(self, index):
        data = self.meta_data[index]
        img0 = np.frombuffer(data[0], dtype='uint8').reshape(256, 448, 3)
        gt = np.frombuffer(data[1], dtype='uint8').reshape(256, 448, 3)
        img1 = np.frombuffer(data[2], dtype='uint8').reshape(256, 448, 3)
        flow_gt = np.frombuffer(self.nf.get(self.flow_data[index]), dtype='float32').reshape(256, 448, 4)
        return img0, gt, img1, flow_gt
            
    def __getitem__(self, index):        
        img0, gt, img1, flow_gt = self.getimg(index)
        if self.dataset_name == 'train':
            '''
            if random.uniform(0, 1) < 0.5:
                aug = iaa.Sequential([
                    iaa.MultiplyHue((0.8, 1.5)),
                    iaa.MultiplyBrightness((0.8, 1.5)),
                    iaa.LinearContrast((0.8, 1.5)),
                ]).to_deterministic()
                img0 = aug(image=img0)
                img1 = aug(image=img1)
                gt = aug(image=gt)
            p1 = random.uniform(1, 1.25)
            p2 = random.uniform(1, 1.25)
            h, w = int(256 * p1), int(448 * p2)
            p1 = h / 256
            p2 = w / 448
            img0 = cv2.resize(img0, (w, h), interpolation=cv2.INTER_LINEAR)
            img1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_LINEAR)
            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_LINEAR)
            flow_gt = cv2.resize(flow_gt, (w, h), interpolation=cv2.INTER_LINEAR)
            flow_gt[:, :, 0] *= p2
            flow_gt[:, :, 1] *= p1
            '''
            img0, gt, img1, flow_gt = self.aug(img0, gt, img1, flow_gt, 224, 224)
            flow_gt = torch.from_numpy(flow_gt.copy()).permute(2, 0, 1)
            img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
            img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
            gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
            return torch.cat((img0, img1, gt), 0), flow_gt                
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
                flow_gt = flow_gt[::-1]
                flow_gt = np.concatenate((flow_gt[:, :, 0:1], -flow_gt[:, :, 1:2], flow_gt[:, :, 2:3], -flow_gt[:, :, 3:4]), 2)
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]
                flow_gt = flow_gt[:, ::-1]
                flow_gt = np.concatenate((-flow_gt[:, :, 0:1], flow_gt[:, :, 1:2], -flow_gt[:, :, 2:3], flow_gt[:, :, 3:4]), 2)
            if random.uniform(0, 1) < 0.5:
                tmp = img1
                img1 = img0
                img0 = tmp
                flow_gt = np.concatenate((flow_gt[:, :, 2:4], flow_gt[:, :, 0:2]), 2)
        else:
            flow_gt = np.zeros((224, 224, 4))
        flow_gt = torch.from_numpy(flow_gt.copy()).permute(2, 0, 1)
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        return torch.cat((img0, img1, gt), 0), flow_gt
