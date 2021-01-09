import runway
import numpy as np
import argparse
import torch
from torchvision import transforms
import os.path
from inference_img import imgint


@runway.command('translate', inputs={'source_imgs': runway.image(description='input image to be translated'),'target': runway.image(description='input image to be translated'),'amount': runway.number(min=0, max=100, default=0)}, outputs={'image': runway.image(description='output image containing the translated result')})
def translate(learn, inputs):
    listimg = imgint(inputs['source_imgs'], inputs['target'])
    i = inputs['amount']
    return (listimg[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]


if __name__ == '__main__':
    runway.run(port=8889)
