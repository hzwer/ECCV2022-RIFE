# RIFE for Nuke

## Introduction

This project implements [**RIFE** - Real-Time Intermediate Flow Estimation for Video Frame Interpolation](https://arxiv.org/abs/2011.06294) for **The Foundry's Nuke**.

**RIFE** is a powerful **frame interpolation** neural network, capable of high-quality retimes and optical flow estimation.

This implementation allows **RIFE** to be used **natively** inside Nuke without any external dependencies or complex installations. It wraps the network in an **easy-to-use Gizmo** with controls similar to those in **OFlow** or **Kronos**.

## Features

- **High quality** retime results competitive with commercial solutions  
- **Fast**, around 0.6 sec/frame for **HD** and 1.5 sec/frame for **4K** on an GeForce RTX 3090
- Support for **RGB**, **alpha**, and **AOVs** channels.
- **Arbitrary timesteps** for animated retimes
- **Downsampling** to reduce memory requirements, allowing **4K/8K** frame sizes.

## Examples

https://github.com/rafaelperez/RIFE-for-Nuke/assets/1684365/6b35a9ea-dee3-414f-9d99-6491ea3c0ff1

https://github.com/rafaelperez/RIFE-for-Nuke/assets/1684365/266f4733-4ed6-4806-accb-ae351d2318da

https://github.com/rafaelperez/RIFE-for-Nuke/assets/1684365/bac1cfd1-4877-438d-bbc8-26cda375dceb

https://github.com/rafaelperez/RIFE-for-Nuke/assets/1684365/6607f72c-1f1e-450d-b15d-d57c2d978bbe

Special thanks to:

- [Blender Studio](https://studio.blender.org) films and assets.
- [ActionVFX](https://www.actionvfx.com) and their [Practice Footage](https://www.actionvfx.com/practice-footage) elements.
- [FXElements](https://www.fxelements.com) and their [Free VFX](https://www.fxelements.com/free) elements.

## Compatibility

**Nuke 14.1+**, tested on **Linux** and **Windows**.

## Installation

1. Download and unzip the latest release from [here](https://github.com/rafaelperez/RIFE-for-Nuke/releases).
2. Copy the extracted `Cattery` folder to `.nuke` or your plugins path.
3. In the toolbar, choose **Cattery > Update** or simply **restart** Nuke.

**RIFE** will then be accessible under the toolbar at **Cattery > Optical Flow > RIFE**.

![cattery_menu_2](https://github.com/rafaelperez/ECCV2022-RIFE/assets/1684365/89239c18-3288-461d-815c-501fd4a63720)

## Options

![rife_nuke](https://github.com/rafaelperez/ECCV2022-RIFE/assets/1684365/4e26d600-8639-41c5-8f52-005deeae2ca2)

- **Input Range:** Defines the frame range for speed calculation, automatically setting to the first frame of the **source** clip.

- **Reset:** Resets the range based on the connected clip.

- **Channels:** Selects the channels for processing - **RGB**, **RGB + Alpha** or **All**.

- **Timing:** Determines the method for retiming:
  - **Speed:** Specifies the retiming in terms of relative duration.
  - **Frame:** Animate destination frame directly.

- **Speed:** Values below 1 decelerate the clip, and above 1, accelerate it.

- **Frame:** Indicates the source frame at the current timeline frame.

- **Output Frame:** Shows the computed output frame, useful for troubleshooting.

- **Downrez:** Reduces the input size to optimize optical flow calculation, lowering memory use. For certain 4K scenes with considerable motion, this preprocessing step can also enhance retime quality.

- **Detail:** Adjusts the processing resolution for the optical flow model. The **maximum** value is **4**. Higher values capture finer movements but also consume more memory. Suggested settings include:
  - **HD:** 3 (± 1)
  - **UHD/4K:** 2 (± 1)

- **Filter:** Filtering for STMap distortion. *Only applies if **Channels** is set to **all***

- **Process only intermediate frames:** When processing on keyframes, RIFE can introduce slight distortion or filtering. This option skips keyframes so they match the original frames exactly.

## Model

**RIFE.cat** uses the latest model from [Practical RIFE](https://github.com/hzwer/Practical-RIFE), version **v4.14** (2024.01.08).

The principal model **IFNet** has been modified for compatibility with **TorchScript**, allowing the model to be compiled into a **.cat** file `(./nuke/Cattery/RIFE/RIFE.cat)`.

This makes it transformable into a native [Nuke's inference node](https://learn.foundry.com/nuke/content/reference_guide/air_nodes/inference.html) through the [CatFileCreator](https://learn.foundry.com/nuke/content/reference_guide/air_nodes/catfilecreator.html).

For more detailed information about the training data and technical specifics, please consult the original repository.

## Compiling the model (Google Colab)

This [Google Colab](https://colab.research.google.com/drive/10TDRhwYiC9-pmNzi97BjVHFj9-br_GZ6) sets up a Linux environment on Google's servers that you can use to compile the TorchScript `RIFE.pt` model.  
This way you don't have to worry about fighting with dependencies on your local machine.


## License and Acknowledgments

**RIFE.cat** is licensed under the MIT License, and is derived from https://github.com/megvii-research/ECCV2022-RIFE.

While the MIT License permits commercial use of RIFE, the dataset used for its training may be under a non-commercial license.

This license does not cover the underlying pre-trained model, associated training data, and dependencies, which may be subject to further usage restrictions.

Consult https://github.com/megvii-research/ECCV2022-RIFE and https://github.com/hzwer/Practical-RIFE for more information on associated licensing terms.

**Users are solely responsible for ensuring that the underlying model, training data, and dependencies align with their intended usage of RIFE.cat.**

## Citation

```
@inproceedings{huang2022rife,
  title={Real-Time Intermediate Flow Estimation for Video Frame Interpolation},
  author={Huang, Zhewei and Zhang, Tianyuan and Heng, Wen and Shi, Boxin and Zhou, Shuchang},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}
```
