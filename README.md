# RIFE
**We are working on arranging our training code and other scripts.**

## Abstract
We propose a real-time intermediate flow estimation algorithm (RIFE) for video frame interpolation (VFI). Most existing methods first estimate the bi-directional optical flows, and then linearly combine them to approximate intermediate flows, leading to artifacts around motion boundaries. We design an intermediate flow model named IFNet that can directly estimate the intermediate flows from coarse to fine. We then warp the input frames according to the estimated intermediate flows and employ a fusion process to compute final results. Based on our proposed leakage distillation, RIFE can be trained end-to-end and achieve excellent performance. Experiments demonstrate that RIFE is significantly faster than existing flow-based VFI methods and achieves state-of-the-art index on several benchmarks.

## Dependencies
```
pip3 install torch==1.6.0
pip3 install numpy
pip3 install opencv-python
```

## Inference and Testing
* Download the pretrained models from [here](https://drive.google.com/file/d/1c1R7iF-ypN6USo-D2YH_ORtaH3tukSlo/view?usp=sharing)

(We also provide 百度网盘 source. 链接: https://pan.baidu.com/s/17tHd-syovvRGP2C6UVPsIw 提取码: 5ha7)
* Unzip and move the pretrained parameters to train_log/\*.pkl

## Usage
```
$ python3 inference.py --img /path/to/image_0 /path/to/image_1
// You will get an out.png. 
```
