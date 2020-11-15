# RIFE
## [arXiv](https://arxiv.org/pdf/2011.06294.pdf) | [Reddit](https://www.reddit.com/r/MachineLearning/comments/jtl37r/r_realtime_720p_flowbased_video_frame/)

**You can easily use [colaboratory](https://colab.research.google.com/github/hzwer/arXiv2020-RIFE/blob/main/Video_2x.ipynb) to have a try.**

Our model can run 30+FPS for 2X 720p interpolation on a 2080Ti GPU. We also support CPU device including MacOS system. Currently our method supports multi-frame interpolation between a pair of images and 2X interpolation for video. Everyone is welcome to use this alpha version and make suggestions!

16X interpolation results from two input images: 

![Demo](./demo/I0_slomo_clipped.gif)
![Demo](./demo/I2_slomo_clipped.gif)

## Abstract
We propose a real-time intermediate flow estimation algorithm (RIFE) for video frame interpolation (VFI). Most existing methods first estimate the bi-directional optical flows, and then linearly combine them to approximate intermediate flows, leading to artifacts around motion boundaries. We design an intermediate flow model named IFNet that can directly estimate the intermediate flows from coarse to fine. We then warp the input frames according to the estimated intermediate flows and employ a fusion process to compute final results. Based on our proposed leakage distillation, RIFE can be trained end-to-end and achieve excellent performance. Experiments demonstrate that RIFE is significantly faster than existing flow-based VFI methods and achieves state-of-the-art index on several benchmarks.

## Dependencies
```
$ pip3 install tqdm
$ pip3 install torch
$ pip3 install numpy
$ pip3 install opencv-python
```
## Usage
* Download the pretrained models from [here](https://drive.google.com/file/d/1c1R7iF-ypN6USo-D2YH_ORtaH3tukSlo/view?usp=sharing)

(We also provide 百度网盘 source. 链接: https://pan.baidu.com/s/17tHd-syovvRGP2C6UVPsIw 提取码: 5ha7)
* Unzip and move the pretrained parameters to train_log/\*.pkl

The models under different setting is coming soon.
**Image Interpolation**

```
$ python3 inference_img.py --img img0.png img1.png --times=4
(2^4=16X interpolation results)
$ ffmpeg -r 10 -f image2 -i output/img%d.png -s 448x256 -c:v libx264 -pix_fmt yuv420p output/slomo.mp4 -q:v 0 -q:a 0
(generate a slomo mp4 video based on two input images)
$ ffmpeg -r 10 -f image2 -i output/img%d.png -s 448x256 -vf "split[s0][s1];[s0]palettegen=stats_mode=single[p];[s1][p]paletteuse=new=1" output/slomo.gif
```

**Video 2x Interpolation**

```
$ python3 inference_mp4_2x.py --video video.mp4
(generate video_2x.mp4, you can use this script recursively to get 4X, 8X, 16X...)
$ python3 inference_mp4_2x.py --video myvideo.mp4 --montage
(if you want to montage the origin video)
```

We will release our training and benchmark validation code soon.

## Evaluation
Download [Vimeo90K dataset](http://toflow.csail.mit.edu/) at ./vimeo_interp_test
```
$ python3 Vimeo90K_benchmark.py
(You will get 35.695PSNR and 0.9788SSIM)
```

## Citation

```
@article{huang2020rife,
  title={RIFE: Real-Time Intermediate Flow Estimation for Video Frame Interpolation},
  author={Huang, Zhewei and Zhang, Tianyuan and Heng, Wen and Shi, Boxin and Zhou, Shuchang},
  journal={arXiv preprint arXiv:2011.06294},
  year={2020}
}
```

## Reference
Optical Flow:
[ARFlow](https://github.com/lliuz/ARFlow)  [pytorch-liteflownet](https://github.com/sniklaus/pytorch-liteflownet)  [RAFT](https://github.com/princeton-vl/RAFT)

Video Interpolation:
[DAIN](https://github.com/baowenbo/DAIN)  [CAIN](https://github.com/myungsub/CAIN)   [AdaCoF-pytorch](https://github.com/HyeongminLEE/AdaCoF-pytorch)
