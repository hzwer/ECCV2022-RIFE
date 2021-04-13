# RIFE v2.4 - Real Time Video Interpolation
## [arXiv](https://arxiv.org/abs/2011.06294) | [YouTube](https://www.youtube.com/watch?v=60DX2T3zyVo&feature=youtu.be) | [Colab](https://colab.research.google.com/github/hzwer/arXiv2020-RIFE/blob/main/Colab_demo.ipynb) | [Tutorial](https://www.youtube.com/watch?v=gf_on-dbwyU&feature=emb_title) | [Demo](https://www.youtube.com/watch?v=oFnyq-e_b3g)

## Table of Contents
1. [Introduction](#introduction)
1. [Collection](#collection)
1. [Usage](#usage)
1. [Evaluation](#evaluation)
1. [Training and Reproduction](#training-and-reproduction)
1. [Citation](#citation)
1. [Reference](#reference)
1. [Sponsor](#sponsor)


## Introduction

2021.3.9 News: We have updated our arXiv paper and fixed most of the defects.

2021.2.9 News: We have updated the RIFEv2 model, faster and much better! Please check our [Update Log](https://github.com/hzwer/arXiv2020-RIFE/issues/41#issuecomment-737651979).

Our model can run 30+FPS for 2X 720p interpolation on a 2080Ti GPU. Currently, our method supports 2X,4X,8X... interpolation, and multi-frame interpolation between a pair of images. Everyone is welcome to use our alpha version and make suggestions!

16X interpolation results from two input images: 

![Demo](./demo/I0_slomo_clipped.gif)
![Demo](./demo/I2_slomo_clipped.gif)

## Collection
**Software**
[Squirrel-RIFE(中文软件)](https://github.com/YiWeiHuang-stack/Squirrel-Video-Frame-Interpolation) | [Waifu2x-Extension-GUI](https://github.com/AaronFeng753/Waifu2x-Extension-GUI) | [Flowframes](https://nmkd.itch.io/flowframes) | [RIFE-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan) | [RIFE-App(Paid)](https://grisk.itch.io/rife-app) | [Autodesk Flame](https://vimeo.com/505942142) |

**2d Animation**
[DAIN-App vs RIFE-App](https://www.youtube.com/watch?v=0OXzVGLRhK0) | [Chika Dance](https://www.youtube.com/watch?v=yjMYefRXikI) | [御坂大哥想让我表白 - 魔女之旅](https://www.bilibili.com/video/BV1r54y1Y7fn) | [ablyh - 超电磁炮](https://www.bilibili.com/video/BV1gK4y1Q7d9?from=search&seid=16584204362417247463) [超电磁炮.b](https://www.bilibili.com/video/BV1sp4y1p73K?from=search&seid=7774996509988438677) | [赫萝与罗伦斯的旅途 - 绫波丽](https://www.bilibili.com/video/BV1yz4y1m7iF) | [花儿不哭 - 乐正绫](https://www.bilibili.com/video/BV1cs411b7qL?from=search&seid=11977379973861203602) |

[没有鼠鼠的雏子Official - 千恋万花](https://www.bilibili.com/video/BV1AT4y1P7kY?from=search&seid=15458655842150253738) | [晨曦光晖 - 从零开始的异世界生活](https://www.bilibili.com/video/BV1QV411i7B4?from=search&seid=151780224584608151) | [琴乃乃 - 天才麻将少女](https://www.bilibili.com/video/BV1Qz4y1y7tH) |

**3d Animation**
[没有鼠鼠的雏子Official - 原神](https://www.bilibili.com/video/BV1iU4y1s7Lk) | [今天我练出腹肌了吗 - 最终幻想](https://www.bilibili.com/video/BV1R541177qr) [仙剑奇侠传](https://www.bilibili.com/video/BV14p4y1s7na) | [娜不列颠 - 冰雪奇缘](https://www.bilibili.com/video/BV1fy4y1J7Mu) | 

**MV and Film**
[Navetek - 邓丽君](https://www.bilibili.com/video/BV1ZK411u7CM) | [生米阿怪 - 周深](https://www.bilibili.com/video/BV1cp4y1W717) | [EzioAuditoreDFirenZe - 中森明菜](https://www.bilibili.com/video/BV1cs411b7qL?from=search&seid=11977379973861203602) | [Life in a Day 2020](https://www.youtube.com/user/lifeinaday) |

**MMD**
[深邃黑暗の银鳕鱼 - 镜音铃](https://www.bilibili.com/video/BV1nU4y1W7RF?from=search&seid=151780224584608151) [fufu](https://www.bilibili.com/video/BV16K4y1Q7CM) [fufu.b](https://www.bilibili.com/video/BV1Xb4y1R7iT) | [Abism0 - 弱音](https://www.bilibili.com/video/BV1Wf4y147cP?from=search&seid=7774996509988438677) |

## Usage

### Installation

```
git clone git@github.com:hzwer/arXiv2020-RIFE.git
cd arXiv2020-RIFE
pip3 install -r requirements.txt
```

* Download the pretrained **HDv2** models from [here](https://drive.google.com/file/d/1wsQIhHZ3Eg4_AfCXItFKqqyDMB4NS0Yd/view?usp=sharing). (百度网盘链接:https://pan.baidu.com/s/1EvQCA8Y7LuDe9KE6XICpXA 密码:xm3z，把压缩包解开后放在 train_log/\*.pkl)

* Unzip and move the pretrained parameters to train_log/\*.pkl

### Run

**Video Frame Interpolation**

You can use our [demo video](https://drive.google.com/file/d/1i3xlKb7ax7Y70khcTcuePi6E7crO_dFc/view?usp=sharing) or your own video. 
```
python3 inference_video.py --exp=1 --video=video.mp4 
```
(generate video_2X_xxfps.mp4)
```
python3 inference_video.py --exp=2 --video=video.mp4
```
(for 4X interpolation)
```
python3 inference_video.py --exp=1 --video=video.mp4 --scale=0.5
```
(If your video has very high resolution such as 4K, we recommend set --scale=0.5 (default 1.0). If you generate disordered pattern on your videos, try set --scale=2.0. This parameter control the process resolution for optical flow model.)
```
python3 inference_video.py --exp=2 --img=input/
```
(to read video from pngs, like input/0.png ... input/612.png, ensure that the png names are numbers)
```
python3 inference_video.py --exp=2 --video=video.mp4 --fps=60
```
(add slomo effect, the audio will be removed)
```
python3 inference_video.py --video=video.mp4 --montage --png
```
(if you want to montage the origin video, skip static frames and save the png format output)

The warning info, 'Warning: Your video has *** static frames, it may change the duration of the generated video.' means that your video has changed the frame rate by adding static frames, it is common if you have processed 25FPS video to 30FPS.

**Image Interpolation**

```
python3 inference_img.py --img img0.png img1.png --exp=4
```
(2^4=16X interpolation results)
After that, you can use pngs to generate mp4:
```
ffmpeg -r 10 -f image2 -i output/img%d.png -s 448x256 -c:v libx264 -pix_fmt yuv420p output/slomo.mp4 -q:v 0 -q:a 0
```
You can also use pngs to generate gif:
```
ffmpeg -r 10 -f image2 -i output/img%d.png -s 448x256 -vf "split[s0][s1];[s0]palettegen=stats_mode=single[p];[s1][p]paletteuse=new=1" output/slomo.gif
```

### Run in docker
Place the pre-trained models in `train_log/\*.pkl` (as above)

Building the container:
```
docker build -t rife -f docker/Dockerfile .
```

Running the container:
```
docker run --rm -it -v $PWD:/host rife:latest inference_video --exp=1 --video=untitled.mp4 --output=untitled_rife.mp4
```
```
docker run --rm -it -v $PWD:/host rife:latest inference_img --img img0.png img1.png --exp=4
```

Using gpu acceleration (requires proper gpu drivers for docker):
```
docker run --rm -it --gpus all -v /dev/dri:/dev/dri -v $PWD:/host rife:latest inference_video --exp=1 --video=untitled.mp4 --output=untitled_rife.mp4
```

## Evaluation
Download [RIFE model](https://drive.google.com/file/d/1U2AGFY00hafsPmm94-6deeM-9feGN-qg/view?usp=sharing) reported by our paper.

**UCF101**: Download [UCF101 dataset](https://liuziwei7.github.io/projects/VoxelFlow) at ./UCF101/ucf101_interp_ours/

**Vimeo90K**: Download [Vimeo90K dataset](http://toflow.csail.mit.edu/) at ./vimeo_interp_test

**MiddleBury**: Download [MiddleBury OTHER dataset](https://vision.middlebury.edu/flow/data/) at ./other-data and ./other-gt-interp

**HD**: Download [HD dataset](https://github.com/baowenbo/MEMC-Net) at ./HD_dataset. We also provide a [google drive download link](https://drive.google.com/file/d/1iHaLoR2g1-FLgr9MEv51NH_KQYMYz-FA/view?usp=sharing).
```
python3 benchmark/UCF101.py
# "PSNR: 35.246 SSIM: 0.9691"
python3 benchmark/Vimeo90K.py
# "PSNR: 35.506 SSIM: 0.9779"
python3 benchmark/MiddleBury_Other.py
# "IE: 1.962"
python3 benchmark/HD.py
# "PSNR: 31.99"
python3 benchmark/HD_multi.py
# "PSNR: 18.89(544*1280), 28.83(720p), 24.96(1080p)"
```

## Training and Reproduction
Because Vimeo90K dataset and the corresponding optical flow labels are too large, we cannot provide a complete dataset download link. We provide you with [a subset containing 100 samples](https://drive.google.com/file/d/1_MQmFWqaptBuEbsV2tmbqFsxmxMIqYDU/view?usp=sharing) for testing the pipeline. Please unzip it at ./dataset

Each sample includes images (I0 I1 Imid : 9 x 256 x 448), and optical flow (flow_t0, flow_t1: 4, 256, 448). 

For origin images, you can download them from [Vimeo90K dataset](http://toflow.csail.mit.edu/).

For generating optical flow labels, our paper use [pytorch-liteflownet](https://github.com/sniklaus/pytorch-liteflownet). In our trainning, we use the optical flow labels generated on 2X size images using official released model. You can also generate labels during training, or [finetune the optical flow network](https://github.com/hzwer/arXiv2020-RIFE/issues/99#issuecomment-798231465) on your training set. 

We use 16 CPUs, 4 GPUs and 20G memory for training: 
```
python3 -m torch.distributed.launch --nproc_per_node=4 train.py --world_size=4
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
[ARFlow](https://github.com/lliuz/ARFlow)  [pytorch-liteflownet](https://github.com/sniklaus/pytorch-liteflownet)  [RAFT](https://github.com/princeton-vl/RAFT)  [pytorch-PWCNet](https://github.com/sniklaus/pytorch-pwc)

Video Interpolation: 
[DVF](https://github.com/lxx1991/pytorch-voxel-flow)  [TOflow](https://github.com/Coldog2333/pytoflow)  [SepConv](https://github.com/sniklaus/sepconv-slomo)  [DAIN](https://github.com/baowenbo/DAIN)  [CAIN](https://github.com/myungsub/CAIN)  [MEMC-Net](https://github.com/baowenbo/MEMC-Net)   [SoftSplat](https://github.com/sniklaus/softmax-splatting)  [BMBC](https://github.com/JunHeum/BMBC)  [EDSC](https://github.com/Xianhang/EDSC-pytorch)  [EQVI](https://github.com/lyh-18/EQVI)

## Sponsor

感谢支持 Paypal Sponsor: https://www.paypal.com/paypalme/hzwer

<img width="160" alt="image" src="https://cdn.luogu.com.cn/upload/image_hosting/5h3609p1.png"><img width="160" alt="image" src="https://cdn.luogu.com.cn/upload/image_hosting/yi3kcwnw.png">
