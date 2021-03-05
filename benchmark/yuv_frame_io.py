#!/usr/bin/env python2.7

import sys
import getopt
import math
import numpy
# import torch
# import torch.utils.serialization
# import PIL
# import PIL.Image

import random
import logging
import numpy as np
from scipy.misc import imsave, imresize
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
# plt.style.use('bmh')
from skimage.color import rgb2yuv, yuv2rgb

import os
from scipy.misc import imread, imsave, imshow, imresize, imsave
from shutil import copyfile
from skimage.measure import compare_ssim, compare_psnr


# from PYTHON_Flow2Color.flowToColor import flowToColor
# from PYTHON_Flow2Color.writeFlowFile import writeFlowFile
# from AverageMeter import  *

class YUV_Read():
    def __init__(self, filepath, h, w, format='yuv420', toRGB=True):

        self.h = h
        self.w = w

        self.fp = open(filepath, 'rb')

        if format == 'yuv420':
            self.frame_length = int(1.5 * h * w)
            self.Y_length = h * w
            self.Uv_length = int(0.25 * h * w)
        else:
            pass
        self.toRGB = toRGB

    def read(self, offset_frame=None):
        if not offset_frame == None:
            self.fp.seek(offset_frame * self.frame_length, 0)

        Y = np.fromfile(self.fp, np.uint8, count=self.Y_length)
        U = np.fromfile(self.fp, np.uint8, count=self.Uv_length)
        V = np.fromfile(self.fp, np.uint8, count=self.Uv_length)
        if Y.size < self.Y_length or \
                        U.size < self.Uv_length or \
                        V.size < self.Uv_length:
            return None, False

        Y = np.reshape(Y, [self.w, self.h], order='F')
        Y = np.transpose(Y)

        U = np.reshape(U, [int(self.w / 2), int(self.h / 2)], order='F')
        U = np.transpose(U)

        V = np.reshape(V, [int(self.w / 2), int(self.h / 2)], order='F')
        V = np.transpose(V)

        U = imresize(U, [self.h, self.w], interp='nearest')
        V = imresize(V, [self.h, self.w], interp='nearest')

        # plt.figure(3)
        # plt.title("GT")
        # plt.imshow(np.stack((Y,Y,Y),axis=-1))
        # plt.show()
        #
        # plt.figure(3)
        # plt.title("GT")
        # plt.imshow(np.stack((U,U,U),axis=-1))
        # plt.show()
        # plt.figure(3)
        # plt.title("GT")
        # plt.imshow(np.stack((V,V,V),axis=-1))
        # plt.show()
        if self.toRGB:
            Y = Y / 255.0
            U = U / 255.0 - 0.5
            V = V / 255.0 - 0.5
            self.YUV = np.stack((Y, U, V), axis=-1)
            self.RGB = (255.0 * np.clip(yuv2rgb(self.YUV), 0.0, 1.0)).astype('uint8')

            # plt.figure(3)
            # plt.title("GT")
            # plt.imshow(self.RGB)
            # plt.show()
            self.YUV = None
            return self.RGB, True
        else:
            self.YUV = np.stack((Y, U, V), axis=-1)
            return self.YUV, True

    def close(self):
        self.fp.close()


class YUV_Write():
    def __init__(self, filepath, fromRGB=True):

        # self.h = h
        # self.w = w
        if os.path.exists(filepath):
            print(filepath)
            # raise("YUV File Exist Error")
  
        self.fp = open(filepath, 'wb')  # no appending
        self.fromRGB = fromRGB

    def write(self, Frame):

        self.h = Frame.shape[0]
        self.w = Frame.shape[1]
        c = Frame.shape[2]

        assert c == 3
        if format == 'yuv420':
            self.frame_length = int(1.5 * self.h * self.w)
            self.Y_length = self.h * self.w
            self.Uv_length = int(0.25 * self.h * self.w)
        else:
            pass
        if self.fromRGB:
            Frame = Frame / 255.0
            YUV = rgb2yuv(Frame)
            Y, U, V = np.dsplit(YUV, 3)
            Y = Y[:, :, 0]
            U = U[:, :, 0]
            V = V[:, :, 0]
            # print(Y.shape)

            # Y = np.transpose(Y,())
            U = np.clip(U + 0.5, 0.0, 1.0)
            V = np.clip(V + 0.5, 0.0, 1.0)

            U = U[::2, ::2]  # imresize(U,[int(self.h/2),int(self.w/2)],interp = 'nearest')
            V = V[::2, ::2]  # imresize(V ,[int(self.h/2),int(self.w/2)],interp = 'nearest')
            Y = (255.0 * Y).astype('uint8')
            U = (255.0 * U).astype('uint8')
            V = (255.0 * V).astype('uint8')
        else:
            YUV = Frame
            Y = YUV[:, :, 0]
            U = YUV[::2, ::2, 1]
            V = YUV[::2, ::2, 2]

        # plt.figure(3)
        # plt.title("GT")
        # plt.imshow(np.stack((Y,Y,Y),axis=-1))
        # plt.show()
        #
        # plt.figure(3)
        # plt.title("GT")
        # plt.imshow(np.stack((U,U,U),axis=-1))
        # plt.show()
        # plt.figure(3)
        # plt.title("GT")
        # plt.imshow(np.stack((V,V,V),axis=-1))
        # plt.show()
        # print(Y.shape)
        # print(U.shape)
        Y = Y.flatten()  # the first order is 0-dimension so don't need to transpose before flatten
        U = U.flatten()
        V = V.flatten()

        Y.tofile(self.fp)
        U.tofile(self.fp)
        V.tofile(self.fp)

        return True

    def close(self):
        self.fp.close()
