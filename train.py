import os
import cv2
import math
import time
import torch
import numpy as np
import random
import argparse
import torch.distributed as dist

from model.RIFE import Model
from dataset import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000.
    else:
        mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
    return  5e-4 * mul

def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)

def train(model, local_rank):
    log_path = 'train_log'
    if local_rank == 0:
        writer = SummaryWriter(log_path + '/train')
        writer_val = SummaryWriter(log_path + '/validate')
    else:
        writer, writer_val = None, None
    step = 0
    nr_eval = 0
    dataset = VimeoDataset('train')
    sampler = DistributedSampler(dataset)
    train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True, sampler=sampler)
    args.step_per_epoch = train_data.__len__()
    dataset_val = VimeoDataset('validation')
    val_data = DataLoader(dataset_val, batch_size=16, pin_memory=True, num_workers=8)
    evaluate(model, val_data, nr_eval, local_rank, writer_val)
    model.save_model(log_path, local_rank)
    print('training...')
    time_stamp = time.time()
    for epoch in range(args.epoch):
        sampler.set_epoch(epoch)
        for i, data in enumerate(train_data):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            data_gpu, flow_gt = data
            data_gpu = data_gpu.to(device, non_blocking=True) / 255.
            flow_gt = flow_gt.to(device, non_blocking=True)
            imgs = data_gpu[:, :6]
            gt = data_gpu[:, 6:9]
            mul = np.cos(step / (args.epoch * args.step_per_epoch) * math.pi) * 0.5 + 0.5
            learning_rate = get_learning_rate(step)
            pred, merged_img, flow, loss_l1, loss_flow, loss_cons, loss_ter, flow_mask = model.update(imgs, gt, learning_rate, mul, True, flow_gt)
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            if step % 100 == 1 and local_rank == 0:
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss_l1', loss_l1, step)
                writer.add_scalar('loss_flow', loss_flow, step)
                writer.add_scalar('loss_cons', loss_cons, step)
                writer.add_scalar('loss_ter', loss_ter, step)
            if step % 1000 == 1 and local_rank == 0:
                gt = (gt.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                pred = (pred.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                merged_img = (merged_img.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                flow = flow.permute(0, 2, 3, 1).detach().cpu().numpy()
                flow_mask = flow_mask.permute(0, 2, 3, 1).detach().cpu().numpy()
                flow_gt = flow_gt.permute(0, 2, 3, 1).detach().cpu().numpy()
                for i in range(5):
                    imgs = np.concatenate((merged_img[i], pred[i], gt[i]), 1)[:, :, ::-1]
                    writer.add_image(str(i) + '/img', imgs, step, dataformats='HWC')
                    writer.add_image(str(i) + '/flow', flow2rgb(flow[i]), step, dataformats='HWC')
                    writer.add_image(str(i) + '/flow_gt', flow2rgb(flow_gt[i]), step, dataformats='HWC')
                    writer.add_image(str(i) + '/flow_mask', flow2rgb(flow[i] * flow_mask[i]), step, dataformats='HWC')
                writer.flush()
            if local_rank == 0:
                print('epoch:{} {}/{} time:{:.2f}+{:.2f} loss_l1:{:.4e}'.format(epoch, i, args.step_per_epoch, data_time_interval, train_time_interval, loss_l1))
            step += 1
        nr_eval += 1
        if nr_eval % 5 == 0:
            evaluate(model, val_data, step, local_rank, writer_val)
        model.save_model(log_path, local_rank)    
        dist.barrier()

def evaluate(model, val_data, nr_eval, local_rank, writer_val):
    loss_l1_list = []
    loss_cons_list = []
    loss_ter_list = []
    loss_flow_list = []
    psnr_list = []
    time_stamp = time.time()
    for i, data in enumerate(val_data):
        data_gpu, flow_gt = data
        data_gpu = data_gpu.to(device, non_blocking=True) / 255.
        flow_gt = flow_gt.to(device, non_blocking=True)
        imgs = data_gpu[:, :6]
        gt = data_gpu[:, 6:9]
        with torch.no_grad():
            pred, merged_img, flow, loss_l1, loss_flow, loss_cons, loss_ter, flow_mask = model.update(imgs, gt, training=False)
        loss_l1_list.append(loss_l1.cpu().numpy())
        loss_flow_list.append(loss_flow.cpu().numpy())
        loss_ter_list.append(loss_ter.cpu().numpy())
        loss_cons_list.append(loss_cons.cpu().numpy())
        for j in range(gt.shape[0]):
            psnr = -10 * math.log10(torch.mean((gt[j] - pred[j]) * (gt[j] - pred[j])).cpu().data)
            psnr_list.append(psnr)
        gt = (gt.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        pred = (pred.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        merged_img = (merged_img.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        flow = flow.permute(0, 2, 3, 1).cpu().numpy()
        if i == 0 and local_rank == 0:
            for j in range(5):
                imgs = np.concatenate((merged_img[i], pred[i], gt[i]), 1)[:, :, ::-1]
                writer_val.add_image(str(i) + '/img', imgs.copy(), nr_eval, dataformats='HWC')
                writer_val.add_image(str(i) + '/flow', flow2rgb(flow[i][:, :, ::-1]), nr_eval, dataformats='HWC')
    
    eval_time_interval = time.time() - time_stamp
    if local_rank == 0:
        print('eval time: {}'.format(eval_time_interval)) 
        writer_val.add_scalar('loss_l1', np.array(loss_l1_list).mean(), nr_eval)
        writer_val.add_scalar('loss_flow', np.array(loss_flow_list).mean(), nr_eval)
        writer_val.add_scalar('loss_cons', np.array(loss_cons_list).mean(), nr_eval)
        writer_val.add_scalar('loss_ter', np.array(loss_ter_list).mean(), nr_eval)
        writer_val.add_scalar('psnr', np.array(psnr_list).mean(), nr_eval)
        
if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='slomo')
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--batch_size', default=16, type=int, help='minibatch size')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--world_size', default=4, type=int, help='world size')
    args = parser.parse_args()
    torch.distributed.init_process_group(backend="nccl", world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    model = Model(args.local_rank)
    train(model, args.local_rank)
        
