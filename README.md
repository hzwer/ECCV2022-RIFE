# RIFE Video Frame Interpolation
## [arXiv](https://arxiv.org/abs/2011.06294) | [Project Page](https://rife-vfi.github.io) | [Reddit](https://www.reddit.com/r/MachineLearning/comments/juv419/r_rife_15fps_to_60fps_video_frame_interpolation/) | [YouTube_v1.2](https://youtu.be/LE2Dzl0oMHI)

**11.18 News: Try our batchsize = 4 parallel processing, got 100% speedup for 2X and 4X video frame interpolation!**

Date of recent model update: 2020.11.19, v1.2

**You can easily use [colaboratory](https://colab.research.google.com/github/hzwer/arXiv2020-RIFE/blob/main/Colab_demo.ipynb) to have a try and generate the above youtube demo.**

Our model can run 30+FPS for 2X 720p interpolation on a 2080Ti GPU. Currently our method supports 2X/4X interpolation for video, and multi-frame interpolation between a pair of images. Everyone is welcome to use this alpha version and make suggestions!

16X interpolation results from two input images: 

![Demo](./demo/I0_slomo_clipped.gif)
![Demo](./demo/I2_slomo_clipped.gif)

## Abstract
We propose RIFE, a Real-time Intermediate Flow Estimation algorithm for Video Frame Interpolation (VFI). Most existing methods first estimate the bi-directional optical flows and then linearly combine them to approximate intermediate flows, leading to artifacts on motion boundaries. RIFE uses a neural network named IFNet that can directly estimate the intermediate flows from images. With the more precise flows and our simplified fusion process, RIFE can improve interpolation quality and have much better speed. Based on our proposed leakage distillation loss, RIFE can be trained in an end-to-end fashion. Experiments demonstrate that our method is significantly faster than existing VFI methods and can achieve state-of-the-art performance on public benchmarks. 

## Dependencies
```
$ pip3 install tqdm
$ pip3 install torch
$ pip3 install numpy
$ pip3 install opencv-python
```
## Usage

* Download the pretrained models from [here](https://drive.google.com/file/d/1zYc3PEN4t6GOUoVYJjvcXoMmM3kFDNGS/view?usp=sharing).
We are optimizing the visual effects and will support animation in the future.

(我们也提供了百度网盘链接:https://pan.baidu.com/s/1YVUsusJFhZ2rWg1Zs5sOkQ  密码:88bu，把压缩包解开后放在 train_log/\*.pkl)
* Unzip and move the pretrained parameters to train_log/\*.pkl

The models under different setting is coming soon.

**Video 2x Interpolation**

You can use our [demo video](https://drive.google.com/file/d/1i3xlKb7ax7Y70khcTcuePi6E7crO_dFc/view?usp=sharing) or use your own video to process. 
```
$ python3 inference_video.py --exp=1 --video=video.mp4 
```
(generate video_2X_xxfps.mp4, you can use this script repeatly to get 4X, 8X...)
```
$ python3 inference_video.py --exp=2 --video=video.mp4
```
(we specificly support exp=2 for 4X interpolation)
```
$ python3 inference_video.py --exp=2 --video=video.mp4 --fps=60
```
(add slomo effect)
```
$ python3 inference_video.py --video=video.mp4 --montage --png
```
(if you want to montage the origin video, and save the png format output)
```
$ python3 inference_video_parallel.py --exp=2 --video=video.mp4
```
(Try our parallel process to get 100% speedup!)

The warning info, 'Warning: Your video has *** static frames, it may change the duration of the generated video.' means that your video has changed the frame rate by adding static frames, it is common if you have processed 25FPS video to 30FPS.

**Image Interpolation**

```
$ python3 inference_img.py --img img0.png img1.png --exp=4
```
(2^4=16X interpolation results)
After that, you can use pngs to generate mp4:
```
$ ffmpeg -r 10 -f image2 -i output/img%d.png -s 448x256 -c:v libx264 -pix_fmt yuv420p output/slomo.mp4 -q:v 0 -q:a 0
```
You can also use pngs to generate gif:
```
$ ffmpeg -r 10 -f image2 -i output/img%d.png -s 448x256 -vf "split[s0][s1];[s0]palettegen=stats_mode=single[p];[s1][p]paletteuse=new=1" output/slomo.gif
```

## Evaluation
First you should download [RIFE model reported by our paper](https://drive.google.com/file/d/1c1R7iF-ypN6USo-D2YH_ORtaH3tukSlo/view?usp=sharing).

We will release our training and benchmark validation code soon.

**Vimeo90K**
Download [Vimeo90K dataset](http://toflow.csail.mit.edu/) at ./vimeo_interp_test
```
$ python3 Vimeo90K_benchmark.py
(You will get 35.695PSNR and 0.9788SSIM)
```

## Citation
<img src="demo/intro.png" alt="img" width=350 />

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
