"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import numpy as np
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

from utils import get_config
from trainer import Trainer

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    type=str,
                    default='configs/funit_animals.yaml')
parser.add_argument('--ckpt',
                    type=str,
                    default='pretrained/animal149_gen.pt')
parser.add_argument('--class_image_folder',
                    type=str,
                    default='images/n02138411')
parser.add_argument('--input',
                    type=str,
                    default='images/input_content.jpg')
parser.add_argument('--output',
                    type=str,
                    default='images/output.jpg')
opts = parser.parse_args()
cudnn.benchmark = True
opts.vis = True
config = get_config(opts.config)
config['batch_size'] = 1
config['gpus'] = 1

trainer = Trainer(config)
trainer.cuda()
trainer.load_ckpt(opts.ckpt)
trainer.eval()

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform_list = [transforms.Resize((128, 128))] + transform_list
transform = transforms.Compose(transform_list)

print('Compute average class codes for images in %s' % opts.class_image_folder)
images = os.listdir(opts.class_image_folder)

# get image b
images = os.listdir(opts.class_image_folder)
if len(images) > 1:
    raise IndexError(
        f"To test one shot translation, please provide one image only."
    )
for i, f in enumerate(images):
    fn = os.path.join(opts.class_image_folder, f)
    img = Image.open(fn).convert('RGB')
    content_img_b = transform(img).unsqueeze(0).cuda()
    with torch.no_grad():
        class_code = trainer.model.compute_k_style(content_img_b, 1)

# get image a
image = Image.open(opts.input)
image = image.convert('RGB')
content_img = transform(image).unsqueeze(0)

print('Compute translation for %s' % opts.input)
with torch.no_grad():
    # generate output images
    # output_image = trainer.model.translate_simple(content_img, class_code) # simple forward pass
    output_images = trainer.model.translate_cross(content_img, content_img_b) # cross forward pass

    # output both mixed images and both reconstructed
    keys = ['m1', 'm2', 'r1', 'r2']
    for key in keys:
        output_image = output_images[key]

        image = output_image.detach().cpu().squeeze().numpy()
        image = np.transpose(image, (1, 2, 0))
        image = ((image + 1) * 0.5 * 255.0)
        output_img = Image.fromarray(np.uint8(image))
        save_path = opts.output[:-3] + key + opts.output[-4:]
        print(save_path)
        output_img.save(save_path, 'JPEG', quality=99)
        print('Save output to %s' % save_path)
