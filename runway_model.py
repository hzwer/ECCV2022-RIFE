import runway
import numpy as np
import argparse
import torch
from torchvision import transforms
import os.path
from inference_img import imgint


@runway.command('translate', inputs={'source_imgs': runway.image(description='input image to be translated'),'target': runway.image(description='input image to be translated'),}, outputs={'image': runway.image(description='output image containing the translated result')})
def translate(learn, inputs):
    imgint(inputs['source_imgs'], inputs['target'])
    counter = 0
    path = "./output/img"+counter+".png"
    img = Image.open(open(path, 'rb'))
    return img


if __name__ == '__main__':
    runway.run(port=8889)
